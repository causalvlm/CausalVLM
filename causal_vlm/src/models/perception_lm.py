"""Separate pretrained Perception/Llama backbones with a new PLM-style projector."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class VisionProjector(nn.Module):
    """Randomly initialized Linear–GELU–Linear, then spatial patch pooling.

    Pooling follows Perception-LM: apply the nonlinear projector to every patch
    before adaptive average pooling. Frame boundaries remain explicit throughout.
    """
    def __init__(self, visual_dim, llm_hidden_size, pooling_ratio, frame_batch_size=8,
                 initializer_range=0.02):
        super().__init__()
        if not isinstance(pooling_ratio, int) or isinstance(pooling_ratio, bool) or pooling_ratio < 1:
            raise ValueError("Projector pooling_ratio must be a positive integer")
        if frame_batch_size < 1 or not math.isfinite(initializer_range) or initializer_range <= 0:
            raise ValueError("Invalid projector chunk size or initialization scale")
        self.pooling_ratio = pooling_ratio
        self.frame_batch_size = frame_batch_size
        self.linear_1 = nn.Linear(visual_dim, llm_hidden_size)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        for layer in (self.linear_1, self.linear_2):
            nn.init.normal_(layer.weight, mean=0.0, std=initializer_range)
            nn.init.zeros_(layer.bias)

    def forward(self, patches):
        if patches.ndim != 4 or patches.shape[-1] != self.linear_1.in_features:
            raise ValueError("Projector needs (batch, frames, patches, encoder_width)")
        b, frames, count, _ = patches.shape
        side = math.isqrt(count)
        if not b or not frames or not count or side * side != count or self.pooling_ratio > side:
            raise ValueError("Patch tokens must form a nonempty square grid at least as large as pooling_ratio")
        pooled = []
        for chunk in patches.flatten(0, 1).split(self.frame_batch_size):
            hidden = self.linear_2(self.gelu(self.linear_1(chunk.to(self.linear_1.weight.dtype))))
            grid = hidden.transpose(1, 2).reshape(len(chunk), -1, side, side)
            grid = F.adaptive_avg_pool2d(grid, (side // self.pooling_ratio,) * 2)
            pooled.append(grid.flatten(2).transpose(1, 2))
        return torch.cat(pooled).reshape(b, frames, -1, self.linear_2.out_features)


class PerceptionVideoEncoder(nn.Module):
    """Return all spatial patch tokens, excluding explicitly configured prefix tokens."""
    def __init__(self, vision_model, mean=(0.5,)*3, std=(0.5,)*3,
                 frame_batch_size=8, num_prefix_tokens=1):
        super().__init__()
        if frame_batch_size < 1 or not isinstance(num_prefix_tokens, int) or num_prefix_tokens < 0:
            raise ValueError("Invalid frame batch size or encoder prefix-token count")
        if len(mean) != 3 or len(std) != 3 or any(not math.isfinite(x) for x in (*mean, *std)) or any(s <= 0 for s in std):
            raise ValueError("RGB normalization requires three finite means and positive scales")
        self.vision_model = vision_model
        self.frame_batch_size = frame_batch_size
        self.num_prefix_tokens = num_prefix_tokens
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, frames):
        if frames.ndim != 6 or frames.shape[3] != 3:
            raise ValueError("Frames must have shape (batch, groups, frames, 3, height, width)")
        b, n, t, c, h, w = frames.shape
        if not b or not n or not t:
            raise ValueError("At least one frame per video is required")
        flat = frames.reshape(b*n*t, c, h, w)
        dtype = next(self.vision_model.parameters()).dtype
        features = []
        for chunk in flat.split(self.frame_batch_size):
            pixels = ((chunk - self.image_mean) / self.image_std).to(dtype)
            hidden = self.vision_model(pixel_values=pixels).last_hidden_state
            if hidden.ndim != 3 or hidden.shape[1] <= self.num_prefix_tokens:
                raise ValueError("Encoder must return patch tokens, not a pooled embedding")
            patches = hidden[:, self.num_prefix_tokens:]
            count = patches.shape[1]
            if math.isqrt(count) ** 2 != count:
                raise ValueError("Encoder patches must form a square grid; check num_prefix_tokens and resolution")
            features.append(patches)
        patches = torch.cat(features)
        return patches.reshape(b, n*t, patches.shape[1], patches.shape[2])


class PerceptionLlamaDecoder(nn.Module):
    def __init__(self, config, visual_dim, llama, tokenizer):
        super().__init__()
        cfg = config["model"]["decoder"]
        proj = config["model"]["projector"]
        self.tokenizer = tokenizer
        if self.tokenizer.eos_token_id is None:
            raise ValueError("The Llama tokenizer must define EOS")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llama = llama
        if hasattr(tokenizer, '__len__') and len(tokenizer) > llama.get_input_embeddings().num_embeddings:
            raise ValueError('Tokenizer vocabulary exceeds the pretrained Llama embeddings; supply its matching tokenizer')
        if cfg.get("use_lora", False):
            from peft import LoraConfig, TaskType, get_peft_model
            self.llama = get_peft_model(llama, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=cfg.get("lora_rank", 16),
                lora_alpha=cfg.get("lora_alpha", 32), lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]))
        self.max_length = cfg.get("max_length", 512)
        self.prompt_before_video = cfg["prompt_before_video"]
        self.prompt_after_video = cfg["prompt_after_video"]
        self.visual_proj = VisionProjector(
            visual_dim, self.llama.config.hidden_size, proj["pooling_ratio"],
            frame_batch_size=config["model"]["encoder"].get("frame_batch_size", 8),
            initializer_range=proj.get("initializer_range", 0.02))

    def _prefix(self, video_tokens):
        embed = self.llama.get_input_embeddings()
        if video_tokens.ndim != 4 or video_tokens.shape[-1] != embed.embedding_dim:
            raise ValueError("Decoder needs projected (batch, frames, patches, Llama_width) tokens")
        visual = video_tokens.flatten(1, 2).to(embed.weight.dtype)
        parts = []
        for prompt, special in ((self.prompt_before_video, True), (self.prompt_after_video, False)):
            tokens = self.tokenizer([prompt] * len(visual), return_tensors="pt",
                                    padding=False, add_special_tokens=special)
            parts.append(embed(tokens["input_ids"].to(visual.device)))
        # Equivalent to replacing the native prompt's video placeholders with patches.
        return torch.cat([parts[0], visual, parts[1]], dim=1)

    def _check_context(self, length):
        maximum = getattr(self.llama.config, "max_position_embeddings", None)
        if maximum is not None and length > maximum:
            raise ValueError(f"Video/prompt/caption require {length} tokens, exceeding Llama context {maximum}; adjust frames, resolution, or projector pooling_ratio")

    def forward(self, video_tokens, labels=None, label_attention_mask=None):
        prefix = self._prefix(video_tokens)
        b, length = prefix.shape[:2]
        mask = torch.ones(b, length, dtype=torch.long, device=prefix.device)
        final_labels = None
        if labels is not None:
            labels = labels.to(prefix.device)
            if label_attention_mask is None:
                raise ValueError("Caption attention masks are required to distinguish padding from EOS")
            caption_mask = label_attention_mask.to(prefix.device).bool()
            if labels.ndim != 2 or labels.shape != caption_mask.shape or len(labels) != b:
                raise ValueError("Caption tokens and masks must have matching batch shapes")
            safe_ids = labels.masked_fill(~caption_mask, self.tokenizer.pad_token_id)
            caption_embeds = self.llama.get_input_embeddings()(safe_ids)
            prefix = torch.cat([prefix, caption_embeds], dim=1)
            mask = torch.cat([mask, caption_mask.long()], dim=1)
            final_labels = torch.cat([
                torch.full((b, length), -100, dtype=torch.long, device=prefix.device),
                labels.masked_fill(~caption_mask, -100)], dim=1)
        self._check_context(prefix.shape[1])
        return self.llama(inputs_embeds=prefix, attention_mask=mask, labels=final_labels,
                          use_cache=False, return_dict=True)

    @torch.no_grad()
    def generate(self, video_tokens):
        if self.training:
            raise ValueError("Call model.eval() before generation")
        prefix = self._prefix(video_tokens)
        self._check_context(prefix.shape[1] + self.max_length)
        ids = self.llama.generate(
            inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device),
            max_new_tokens=self.max_length, eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id, do_sample=False, use_cache=True)
        # With inputs_embeds, Transformers returns generated IDs without prompt IDs.
        return [text.strip() for text in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]


class PerceptionVideoLM(nn.Module):
    def __init__(self, config, vision_model, llama, tokenizer):
        super().__init__()
        cfg = config["model"]["encoder"]
        if "token_pooling" in cfg:
            raise ValueError("Encoder configuration requires patch tokens and num_prefix_tokens")
        self.encoder = PerceptionVideoEncoder(
            vision_model, mean=cfg.get("mean", (0.5,)*3), std=cfg.get("std", (0.5,)*3),
            frame_batch_size=cfg.get("frame_batch_size", 8), num_prefix_tokens=cfg["num_prefix_tokens"])
        self.decoder = PerceptionLlamaDecoder(config, cfg["hidden_size"], llama, tokenizer)
        self.visual_hidden_size = self.decoder.llama.config.hidden_size
        dec = config["model"]["decoder"]
        architecture_fields = (
            'model_type', 'architecture', 'model_args', 'hidden_size', 'vocab_size',
            'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
            'intermediate_size', 'head_dim', 'max_position_embeddings', 'rope_theta',
            'rope_scaling', 'hidden_act', 'rms_norm_eps', 'layer_norm_eps',
            'attention_bias', 'attention_dropout', 'tie_word_embeddings', 'patch_size', 'image_size',
        )
        def architecture_of(backbone):
            backbone_config = getattr(backbone, 'config', None)
            return {key: getattr(backbone_config, key) for key in architecture_fields
                    if hasattr(backbone_config, key)}
        self._spec = {
            "architecture": "perception_lm_separate_backbones_v1",
            "encoder_width": cfg["hidden_size"], "language_width": self.visual_hidden_size,
            "num_prefix_tokens": cfg["num_prefix_tokens"],
            "mean": list(cfg.get("mean", (0.5,)*3)), "std": list(cfg.get("std", (0.5,)*3)),
            "pooling_ratio": config["model"]["projector"]["pooling_ratio"],
            "prompt_before_video": dec["prompt_before_video"], "prompt_after_video": dec["prompt_after_video"],
            "use_lora": dec.get("use_lora", False),
            "lora_rank": dec.get("lora_rank", 16) if dec.get("use_lora", False) else None,
            "lora_alpha": dec.get("lora_alpha", 32) if dec.get("use_lora", False) else None,
            "resolution": config["data"]["resolution"], "num_frames": config["data"]["num_frames"],
            "encoder_architecture": architecture_of(vision_model),
            "language_architecture": architecture_of(llama),
        }

    def checkpoint_spec(self):
        from copy import deepcopy
        spec = deepcopy(self._spec)
        spec.update(pooling_ratio=self.decoder.visual_proj.pooling_ratio,
                    num_prefix_tokens=self.encoder.num_prefix_tokens,
                    prompt_before_video=self.decoder.prompt_before_video,
                    prompt_after_video=self.decoder.prompt_after_video)
        return spec

    def encode_video(self, frames):
        """Projected spatial tokens (B, frames, pooled patches, language width)."""
        return self.decoder.visual_proj(self.encoder(frames))

    def forward(self, frames, labels=None, label_attention_mask=None):
        return self.decoder(self.encode_video(frames), labels, label_attention_mask)

    def generate(self, frames):
        return self.decoder.generate(self.encode_video(frames))
