"""Strict checkpoint loading; saves are replaced atomically."""
from pathlib import Path
import os
import torch


def save_checkpoint(state, is_best, checkpoint_dir, filename):
    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [filename, 'last_model.pt'] + (['best_objective.pt'] if is_best else [])
    for name in names:
        temporary = out / (name + '.tmp')
        torch.save(state, temporary)
        os.replace(temporary, out / name)


def model_spec(model):
    captioner = model.stage1 if hasattr(model, 'stage1') else model
    spec = {'captioner': captioner.checkpoint_spec()}
    if hasattr(model, 'causal_head'):
        head = model.causal_head
        spec['causal_head'] = {
            'visual_dim': head.visual_proj[0].in_features,
            'text_dim': head.text_proj[0].in_features,
            'hidden_dim': head.cross_attention.embed_dim,
            'n_heads': head.cross_attention.num_heads,
            'layers': len(head.transformer_encoder.layers),
            'ffn_dim': head.transformer_encoder.layers[0].linear1.out_features,
            'event_pooling': head.event_pooling, 'text_encoding': model.text_encoding,
            'text_max_length': model.text_max_length,
        }
    return spec


def restore_checkpoint(checkpoint, model):
    if checkpoint.get('format_version') != 2:
        raise ValueError('Expected a version-2 training checkpoint; use predict_published for the Hugging Face export')
    if checkpoint.get('model_spec') != model_spec(model):
        raise ValueError('Checkpoint architecture/preprocessing differs from this model; check pooling, prompts, dimensions, and stage')
    state = checkpoint['model_state_dict']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def load_checkpoint(checkpoint_path, model):
    return restore_checkpoint(torch.load(checkpoint_path, map_location='cpu', weights_only=True), model)
