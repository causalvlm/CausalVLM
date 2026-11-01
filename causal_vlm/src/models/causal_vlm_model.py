from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MAX_EVENTS: int = 20
HIDDEN_DIM:   int = 512


class MultimodalCausalHead(nn.Module):
    """
    Stage 2 causal head (Section 3.3, Appendix A).

    Inputs
        V  (B, F, D_v)       frame means of Stage 1 projected patch tokens
        T  (B, N_max, D_t)   per-event means of Stage 1 LLM final hidden states

    Output
        A  (B, N_max, N_max) causal adjacency matrix, upper-triangular, values in [0,1]

    Architecture (paper equations 3, 6-14)
        Eq. 3   V_event = mean of frames within each event interval
        Eq. 6   V' = GELU(LN(W_v · V_event))
        Eq. 7   T' = GELU(LN(W_t · T[:N]))
        Eq. 8   F, _ = MHA(Q=V', K=T', V=T')
        Eq. 9   E = V' + F
        Eq. 10  C = TransformerEncoder(E)          [2 layers, H=512, FFN=1024]
        Eq. 11  P_ij = [C_i; C_j]  in R^{2H}
        Eq. 12  h = GELU(LN(W1·P_ij)), drop=0.2  then GELU(W2·h)
        Eq. 13  z_ij = w3^T · h
        Eq. 14  A_ij = sigmoid(z_ij) if i<j else 0
    """

    def __init__(
        self,
        visual_dim:       int,
        text_dim:         int,
        hidden_dim:       int   = HIDDEN_DIM,
        n_heads:          int   = 8,
        n_encoder_layers: int   = 2,
        ffn_dim:          int   = 1024,
        dropout:          float = 0.1,
        n_max_events:     int   = N_MAX_EVENTS,
        event_pooling: str = "timestamp",
    ) -> None:
        super().__init__()
        self.n_max_events = n_max_events  # Retained for configuration compatibility; event counts remain dynamic.
        if event_pooling not in {"timestamp", "uniform"}:
            raise ValueError("event_pooling must be timestamp or uniform")
        self.event_pooling = event_pooling

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=ffn_dim,
                dropout=dropout, activation="gelu", batch_first=True,
            ),
            num_layers=n_encoder_layers,
        )
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _pool(frame_features: torch.Tensor, n: int) -> torch.Tensor:
        return F.adaptive_avg_pool1d(
            frame_features.T.unsqueeze(0), n
        ).squeeze(0).T
    @staticmethod
    def _pool_ts(V: torch.Tensor, fts: torch.Tensor, ev: torch.Tensor, n: int) -> torch.Tensor:
        """Pool inclusive intervals; use the nearest midpoint frame for empty intervals."""
        if V.ndim != 2 or fts.shape != (V.shape[0],) or not V.shape[0]:
            raise ValueError("Frame timestamps must match the nonempty frame features")
        if ev.ndim != 2 or ev.shape[0] < n or ev.shape[1] != 2:
            raise ValueError("Every event needs [start, end] timestamps")
        if not torch.isfinite(fts).all() or not torch.isfinite(ev[:n]).all():
            raise ValueError("Timestamps must be finite")
        if (ev[:n, 1] <= ev[:n, 0]).any():
            raise ValueError("Event ends must be later than starts")
        if n > 1 and (ev[1:n, 0] < ev[:n-1, 0]).any():
            raise ValueError("Events must be ordered by onset")
        pooled = []
        for start, end in ev[:n]:
            inside = (fts >= start) & (fts <= end)
            pooled.append(V[inside].mean(0) if inside.any()
                          else V[(fts - (start + end) / 2).abs().argmin()])
        return torch.stack(pooled) if pooled else V.new_empty((0, V.shape[-1]))

    def forward(
        self,
        visual_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_events:      List[int],
        frame_ts:        Optional[torch.Tensor] = None,
        event_ts:        Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B      = visual_features.shape[0]
        device = visual_features.device
        dtype  = visual_features.dtype
        num_events = [int(n) for n in num_events]
        if len(num_events) != B or any(n < 0 for n in num_events):
            raise ValueError("One nonnegative event count is required per video")
        N_max = max(num_events, default=0)
        if text_embeddings.shape[0] != B or text_embeddings.shape[1] < N_max:
            raise ValueError("Text embeddings do not match the event counts")
        if self.event_pooling == "timestamp" and N_max and (frame_ts is None or event_ts is None):
            raise ValueError("Timestamp pooling requires frame and event timestamps; no uniform fallback")

        packed = torch.zeros(B, N_max, N_max, device=device, dtype=torch.float32)

        for b in range(B):
            N = num_events[b]
            if N < 1:
                continue

            if self.event_pooling == "timestamp":
                v_event = self._pool_ts(visual_features[b], frame_ts[b].to(device), event_ts[b].to(device), N)
            else:
                v_event = self._pool(visual_features[b], N)
            v_prime = self.visual_proj(v_event)
            t_prime = self.text_proj(text_embeddings[b, :N])

            fused, _ = self.cross_attention(
                v_prime.unsqueeze(0), t_prime.unsqueeze(0), t_prime.unsqueeze(0)
            )
            e = v_prime + fused.squeeze(0)
            c = self.transformer_encoder(e.unsqueeze(0)).squeeze(0)

            c_i    = c.unsqueeze(1).expand(N, N, -1)
            c_j    = c.unsqueeze(0).expand(N, N, -1)
            logits = self.pairwise_mlp(torch.cat([c_i, c_j], dim=-1)).squeeze(-1)

            mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
            adj  = torch.zeros(N, N, device=device, dtype=torch.float32)
            adj[mask] = torch.sigmoid(logits[mask].float())
            packed[b, :N, :N] = adj

        return packed + self.pairwise_mlp[-1].weight.sum() * 0, torch.tensor(num_events, dtype=torch.long, device=device)


class CausalVLM(nn.Module):
    """Captioner and causal head with differentiable, contextual event text encoding."""

    def __init__(self, stage1: nn.Module, causal_head: MultimodalCausalHead,
                 text_encoding: str = "contextual", text_batch_size: int = 8,
                 text_max_length: int = 64) -> None:
        super().__init__()
        if text_encoding not in {"contextual", "embed"}:
            raise ValueError("text_encoding must be contextual or embed")
        if text_batch_size < 1 or text_max_length < 1:
            raise ValueError("Text batch size and maximum length must be positive")
        self.stage1 = stage1
        self.causal_head = causal_head
        self.text_encoding = text_encoding
        self.text_batch_size = text_batch_size
        self.text_max_length = text_max_length

    def encode_event_text(self, event_descriptions: List[List[str]], device) -> torch.Tensor:
        tokenizer = self.stage1.decoder.tokenizer
        llm = self.stage1.decoder.llama
        embed = llm.get_input_embeddings()
        counts = [len(ds) for ds in event_descriptions]
        texts = [text for ds in event_descriptions for text in ds]
        n_max = max(counts, default=0)
        if not texts:
            return embed.weight.new_zeros((len(counts), 0, embed.embedding_dim), device=device)
        encoded = []
        for start in range(0, len(texts), self.text_batch_size):
            tokens = tokenizer(texts[start:start+self.text_batch_size], return_tensors="pt",
                               padding=True, truncation=True, max_length=self.text_max_length)
            ids = tokens["input_ids"].to(device)
            mask = tokens["attention_mask"].to(device)
            if self.text_encoding == "contextual":
                # Preserve gradients into the language model during joint training.
                hidden = llm(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                             use_cache=False, return_dict=True).hidden_states[-1]
            else:
                hidden = embed(ids)
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            encoded.append((hidden * weights).sum(1) / weights.sum(1).clamp_min(1))
        flat = torch.cat(encoded)
        padded, offset = [], 0
        for n in counts:
            padded.append(F.pad(flat[offset:offset+n], (0, 0, 0, n_max-n)))
            offset += n
        return torch.stack(padded)

    def forward(self, frames, event_descriptions=None, num_events=None, labels=None,
                frame_timestamps=None, event_timestamps=None, label_attention_mask=None) -> dict:
        video_tokens = self.stage1.encode_video(frames)
        visual_features = video_tokens.mean(dim=2)
        caption_loss = None
        if labels is not None:
            caption_loss = self.stage1.decoder(video_tokens, labels=labels,
                                               label_attention_mask=label_attention_mask).loss
        adjacency = sizes = None
        if event_descriptions is not None:
            expected = [len(ds) for ds in event_descriptions]
            if num_events is None:
                num_events = expected
            if expected != [int(n) for n in num_events]:
                raise ValueError("Event descriptions and counts must agree")
            text = self.encode_event_text(event_descriptions, frames.device)
            adjacency, sizes = self.causal_head(visual_features, text, num_events,
                                                frame_timestamps, event_timestamps)
        return {"visual_features": visual_features, "caption_loss": caption_loss,
                "adjacency_matrix": adjacency, "sizes": sizes}

    @torch.no_grad()
    def predict(self, frames, frame_timestamps, durations) -> list[dict]:
        """Full pipeline using only generated events and observed video metadata."""
        from ..events import parse_events
        if self.training:
            raise ValueError("Call model.eval() before predict()")
        if len(durations) != len(frames):
            raise ValueError("One video duration is required per input")
        video_tokens = self.stage1.encode_video(frames)
        visual = video_tokens.mean(dim=2)
        captions = self.stage1.decoder.generate(video_tokens)
        events = [parse_events(caption, float(duration)) for caption, duration in zip(captions, durations)]
        descriptions = [[e["description"] for e in ev] for ev in events]
        counts = [len(ev) for ev in events]
        n_max = max(counts, default=0)
        timestamps = visual.new_zeros((len(events), n_max, 2), dtype=torch.float32)
        for b, ev in enumerate(events):
            if ev:
                timestamps[b, :len(ev)] = torch.tensor([e["timestamp"] for e in ev], device=visual.device)
        text = self.encode_event_text(descriptions, visual.device)
        adjacency, _ = self.causal_head(visual, text, counts, frame_timestamps, timestamps)
        return [{"events": ev, "adjacency_matrix": adjacency[b, :len(ev), :len(ev)].cpu().tolist(),
                 "generated_text": captions[b]} for b, ev in enumerate(events)]
