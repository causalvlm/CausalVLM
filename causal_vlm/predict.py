"""Predict event captions, intervals, and forward causal edges from sampled video frames.

Input needs video metadata and sampled frames only, without annotations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .src.models.factory import build_model
from .src.utils.checkpoint import restore_checkpoint


def load_frames(record, base, config):
    paths = record['frame_paths']
    timestamps = torch.tensor(record['frame_timestamps'], dtype=torch.float32)
    duration = float(record['duration'])
    expected = config['data']['num_frames']
    if len(paths) != expected or timestamps.shape != (expected,):
        raise ValueError(f'Input requires {expected} frame paths and matching timestamps')
    if not np.isfinite(duration) or duration <= 0 or not torch.isfinite(timestamps).all():
        raise ValueError('Duration and frame timestamps must be finite')
    if (timestamps < 0).any() or (timestamps > duration).any() or (timestamps[1:] < timestamps[:-1]).any():
        raise ValueError('Frame timestamps must be ordered and within the video')
    resolution = config['data']['resolution']
    images = []
    for path in paths:
        with Image.open(base/path) as image:
            images.append(np.asarray(image.convert('RGB').resize((resolution, resolution), Image.Resampling.BILINEAR)))
    pixels = np.stack(images).astype(np.float32).transpose(0, 3, 1, 2) / 255
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(2), timestamps.unsqueeze(0), [duration]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True, help='JSON mapping video IDs to frame paths, timestamps, duration')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--encoder-checkpoint', help='Override the encoder location saved in the checkpoint')
    p.add_argument('--llm-checkpoint', help='Override the Llama location saved in the checkpoint')
    p.add_argument('--edge-threshold', type=float, default=0.5)
    a = p.parse_args()
    if not 0 <= a.edge_threshold <= 1:
        p.error('Edge threshold must be in [0, 1]')
    output = Path(a.output)
    if output.exists():
        raise FileExistsError('Choose a new prediction output path')
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 2:
        raise ValueError('Inference requires a version-2 separate-backbone checkpoint including its trained projector')
    config, stage = payload['config'], payload['stage']
    if a.encoder_checkpoint:
        config['model']['encoder']['model_name'] = a.encoder_checkpoint
    if a.llm_checkpoint:
        config['model']['decoder']['model_name'] = a.llm_checkpoint
    device = torch.device(a.device)
    model = build_model(config, stage, device)
    restore_checkpoint(payload, model)
    del payload
    model.eval()
    filename = Path(a.input)
    records = json.loads(filename.read_text())
    predictions = {}
    from .src.events import parse_events
    for vid, record in records.items():
        frames, timestamps, durations = load_frames(record, filename.parent, config)
        if stage == 2:
            prediction = model.predict(frames.to(device), timestamps.to(device), durations)[0]
            matrix = prediction['adjacency_matrix']
            prediction['edges'] = [[i+1, j+1] for i, row in enumerate(matrix)
                                    for j, probability in enumerate(row) if probability > a.edge_threshold]
            prediction['edge_threshold'] = a.edge_threshold
        else:
            text = model.generate(frames.to(device))[0]
            prediction = {'events': parse_events(text, durations[0]), 'generated_text': text}
        predictions[vid] = prediction
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(predictions, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(f'Wrote {len(predictions)} predictions to {output}')


if __name__ == '__main__':
    main()
