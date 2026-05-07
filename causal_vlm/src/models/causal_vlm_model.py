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
        V  (B, F, D_v)       frame-level visual features from Stage 1 encoder
        T  (B, N_max, D_t)   per-event text embeddings from Stage 1 LLM embedding layer

    Output
        A  (B, N_max, N_max) causal adjacency matrix, upper-triangular, values in [0,1]

    Architecture (paper equations 3, 6-14)
        Eq. 3   V_event = AdaptiveAvgPool(V^T, N)^T
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
    ) -> None:
        super().__init__()
        self.n_max_events = n_max_events

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

    def forward(
        self,
        visual_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_events:      List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B      = visual_features.shape[0]
        device = visual_features.device
        dtype  = visual_features.dtype
        N_max  = max(num_events) if num_events else 1

        packed = torch.zeros(B, N_max, N_max, device=device, dtype=dtype)

        for b in range(B):
            N = num_events[b]
            if N < 1:
                continue

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
            adj  = torch.zeros(N, N, device=device, dtype=dtype)
            adj[mask] = torch.sigmoid(logits[mask])
            packed[b, :N, :N] = adj

        return packed, torch.tensor(num_events, dtype=torch.long, device=device)


class CausalVLM(nn.Module):
    """
    Full Causal-VLM model: Stage 1 (PerceptionVideoLM) + Stage 2 (MultimodalCausalHead).

    Text embeddings T are produced by tokenising event descriptions and mean-pooling
    the Stage 1 LLM's token embedding layer (Section 3.3).
    D_t = llm.config.hidden_size (3072 for Llama-3.1-3B-Instruct).
    """

    def __init__(self, stage1: nn.Module, causal_head: MultimodalCausalHead) -> None:
        super().__init__()
        self.stage1      = stage1
        self.causal_head = causal_head

    def encode_event_text(
        self,
        event_descriptions: List[List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        tokenizer   = self.stage1.decoder.tokenizer
        embed_layer = self.stage1.decoder.llama.get_input_embeddings()
        D_t         = embed_layer.embedding_dim
        B           = len(event_descriptions)
        N_max       = max(len(d) for d in event_descriptions) if event_descriptions else 1

        out = torch.zeros(B, N_max, D_t, device=device, dtype=torch.bfloat16)
        for b, descs in enumerate(event_descriptions):
            for n, text in enumerate(descs):
                ids = tokenizer(
                    text, return_tensors="pt", padding=False, truncation=True, max_length=64
                )["input_ids"].to(device)
                out[b, n] = embed_layer(ids).mean(dim=1).squeeze(0).to(torch.bfloat16)
        return out

    def forward(
        self,
        frames:             torch.Tensor,
        event_descriptions: Optional[List[List[str]]],
        num_events:         Optional[List[int]],
        labels:             Optional[torch.Tensor] = None,
    ) -> dict:
        device = frames.device

        visual_features = self.stage1.encoder(frames)

        caption_loss = None
        if labels is not None:
            # Reuse the already-computed visual_features to avoid a second encoder pass.
            caption_loss = self.stage1.decoder(visual_features, labels=labels).loss

        adjacency_matrix = sizes = None
        if event_descriptions is not None and num_events is not None:
            T     = self.encode_event_text(event_descriptions, device)
            N_max = max(num_events)
            if T.shape[1] < N_max:
                pad = torch.zeros(T.shape[0], N_max - T.shape[1], T.shape[2],
                                  device=device, dtype=T.dtype)
                T = torch.cat([T, pad], dim=1)
            adjacency_matrix, sizes = self.causal_head(visual_features, T, num_events)

        return {
            "visual_features":  visual_features,
            "caption_loss":     caption_loss,
            "adjacency_matrix": adjacency_matrix,
            "sizes":            sizes,
        }
