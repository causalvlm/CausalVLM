import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class VisionProjector(nn.Module):
    def __init__(self, visual_dim: int, llm_hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(visual_dim, llm_hidden_size)
        self.gelu     = nn.GELU()
        self.linear_2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.gelu(self.linear_1(x)))


class PerceptionVideoEncoder(nn.Module):
    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, N, T, C, H, W = frames.shape
        flat = frames.view(B * N * T, C, H, W)
        with torch.set_grad_enabled(self.vision_model.training):
            out = self.vision_model(pixel_values=flat)
        cls = out.last_hidden_state[:, 0, :]          # CLS token per frame
        return cls.view(B, N * T, -1).to(torch.bfloat16).requires_grad_(True)


class PerceptionLlamaDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        model_name = config.model.decoder.model_name
        visual_dim  = config.model.encoder.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        if config.model.decoder.use_lora:
            self.llama = get_peft_model(
                self.llama,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=config.model.decoder.lora_rank,
                    lora_alpha=config.model.decoder.lora_alpha,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                ),
            )

        self.max_length  = config.model.decoder.max_length
        self.visual_proj = VisionProjector(
            visual_dim, self.llama.config.hidden_size
        ).to(torch.bfloat16)

    def forward(
        self,
        video_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        device = video_features.device
        B      = video_features.shape[0]

        vis_embeds = self.visual_proj(video_features)

        prompt_ids = self.tokenizer(
            ["Describe the following video:"] * B,
            return_tensors="pt",
            padding=False,
        )["input_ids"].to(device)
        prompt_embeds = self.llama.get_input_embeddings()(prompt_ids)

        if labels is not None:
            labels = labels.to(device)
            cap_embeds    = self.llama.get_input_embeddings()(labels)
            inputs_embeds = torch.cat([vis_embeds, prompt_embeds, cap_embeds], dim=1)
            attn_mask     = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

            # Mask visual and prompt tokens from the loss
            vis_len    = vis_embeds.shape[1]
            prompt_len = prompt_embeds.shape[1]
            final_labels = torch.cat([
                torch.full((B, vis_len + prompt_len), -100, device=device),
                labels,
            ], dim=1)

            # Truncate to shortest if length mismatch
            L = min(final_labels.shape[1], inputs_embeds.shape[1])
            inputs_embeds = inputs_embeds[:, :L]
            attn_mask     = attn_mask[:, :L]
            final_labels  = final_labels[:, :L]
        else:
            inputs_embeds = torch.cat([vis_embeds, prompt_embeds], dim=1)
            attn_mask     = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            final_labels  = None

        return self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=final_labels,
            return_dict=True,
        )

    def generate(self, video_features: torch.Tensor) -> list[str]:
        self.llama.eval()
        device = video_features.device
        B      = video_features.shape[0]

        vis_embeds    = self.visual_proj(video_features).to(self.llama.dtype)
        prompt_ids    = self.tokenizer(
            ["Describe the following video:"] * B,
            return_tensors="pt", padding=False,
        )["input_ids"].to(device)
        prompt_embeds = self.llama.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([vis_embeds, prompt_embeds], dim=1)
        attn_mask     = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        with torch.no_grad():
            ids = self.llama.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                max_new_tokens=self.max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
            )
        return [
            t.strip() for t in self.tokenizer.batch_decode(
                ids[:, inputs_embeds.shape[1]:], skip_special_tokens=True
            )
        ]


class PerceptionVideoLM(nn.Module):
    def __init__(self, config, vision_model: nn.Module) -> None:
        super().__init__()
        config.model.encoder.hidden_size = vision_model.config.num_features
        self.encoder = PerceptionVideoEncoder(vision_model)
        self.decoder = PerceptionLlamaDecoder(config)

    def forward(self, frames: torch.Tensor, labels: torch.Tensor | None = None):
        return self.decoder(self.encoder(frames), labels)

    def generate(self, frames: torch.Tensor) -> list[str]:
        return self.decoder.generate(self.encoder(frames))
