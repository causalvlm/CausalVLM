"""Load two independent pretrained backbones; initialize only the new projector/head."""
from __future__ import annotations
from types import SimpleNamespace
import torch
from torch import nn
from .perception_lm import PerceptionVideoLM
from .causal_vlm_model import CausalVLM, MultimodalCausalHead


class TimmEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(hidden_size=model.num_features)

    def forward(self, pixel_values):
        hidden = self.model.forward_features(pixel_values)
        if hidden.ndim != 3:
            raise ValueError('Perception encoder must return patch-token features shaped (B, tokens, D)')
        return SimpleNamespace(last_hidden_state=hidden)


def visual_width(vision, configured=None):
    found = None
    for obj in (getattr(vision, 'config', None), vision):
        for name in ('hidden_size', 'num_features', 'embed_dim'):
            value = getattr(obj, name, None)
            if value is not None:
                found = int(value)
                break
        if found is not None:
            break
    arguments = getattr(getattr(vision, 'config', None), 'model_args', {}) or {}
    if found is None and 'embed_dim' in arguments:
        found = int(arguments['embed_dim'])
    if found is not None and configured is not None and int(configured) != found:
        raise ValueError(f'Configured encoder width {configured} disagrees with loaded width {found}')
    width = found if found is not None else configured
    if width is None or int(width) < 1:
        raise ValueError('Set model.encoder.hidden_size to the actual encoder output width')
    return int(width)


def pretrained(loader, name, **kwargs):
    model, info = loader.from_pretrained(name, output_loading_info=True, **kwargs)
    if any(info.get(key) for key in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Backbone {name} did not load completely: {info}')
    return model


def build_model(config, stage, device):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
    if stage not in (1, 2):
        raise ValueError('Stage must be 1 or 2')
    enc, dec = config['model']['encoder'], config['model']['decoder']
    if not enc.get('model_name') or not dec.get('model_name'):
        raise ValueError('Supply separate --encoder-checkpoint and --llm-checkpoint paths/IDs; no model is substituted')
    if 'projector' not in config['model'] or 'num_prefix_tokens' not in enc or 'token_pooling' in enc:
        raise ValueError('Use the separate-backbone patch-token configuration with an explicit projector pooling_ratio')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    language_config = AutoConfig.from_pretrained(dec['model_name'])
    if language_config.model_type != 'llama':
        raise ValueError('The language checkpoint must be a standalone Llama, not a combined vision-language model')
    if enc.get('loader', 'transformers') == 'timm':
        import timm
        vision = TimmEncoder(timm.create_model(enc['model_name'], pretrained=True, num_classes=0))
    elif enc.get('loader', 'transformers') == 'transformers':
        vision_config = AutoConfig.from_pretrained(enc['model_name'])
        if hasattr(vision_config, 'text_config') or vision_config.model_type in {'perception_lm', 'causalvlm'}:
            raise ValueError('Supply the standalone Perception encoder checkpoint, not a combined vision-language model')
        vision = pretrained(AutoModel, enc['model_name'], config=vision_config, torch_dtype=dtype)
    else:
        raise ValueError('Encoder loader must be transformers or timm')
    enc['hidden_size'] = visual_width(vision, enc.get('hidden_size'))
    timm_model = getattr(vision, 'timm_model', getattr(vision, 'model', None))
    prefix_count = getattr(timm_model, 'num_prefix_tokens', None)
    if prefix_count is not None and prefix_count != enc['num_prefix_tokens']:
        raise ValueError(f'Encoder has {prefix_count} prefix tokens; configure num_prefix_tokens accordingly')
    llm = pretrained(AutoModelForCausalLM, dec['model_name'], config=language_config, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(dec.get('tokenizer_name') or dec['model_name'])
    stage1 = PerceptionVideoLM(config, vision, llm, tokenizer).to(device=device, dtype=dtype)
    if stage == 1:
        return stage1
    head_cfg = config['model']['causal_head']
    head = MultimodalCausalHead(stage1.visual_hidden_size, llm.config.hidden_size,
                               hidden_dim=head_cfg['hidden_dim'], n_heads=head_cfg['n_heads'],
                               n_encoder_layers=head_cfg['layers'], ffn_dim=head_cfg['ffn_dim'],
                               event_pooling=head_cfg['event_pooling']).to(device=device, dtype=dtype)
    return CausalVLM(stage1, head, text_encoding=head_cfg['text_encoding'],
                     text_batch_size=head_cfg['text_batch_size'], text_max_length=head_cfg['text_max_length'])
