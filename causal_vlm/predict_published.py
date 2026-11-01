"""Run Causal-VLM on video files using the published Hugging Face weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time


def verify_files(directory, profile):
    """Check every runtime file before importing the checkpoint's model code."""
    verified = {}
    for name, expected in profile['files'].items():
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size != expected['size']:
            raise ValueError(f'{name}: missing file or size differs from the expected checkpoint')
        if expected['algorithm'] == 'sha256':
            digest = hashlib.sha256()
        elif expected['algorithm'] == 'git-blob-sha1':
            digest = hashlib.sha1(b'blob ' + str(expected['size']).encode() + b'\0')
        else:
            raise ValueError('Unsupported checkpoint hash algorithm')
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected['hash']:
            raise ValueError(f'{name}: contents differ from the expected checkpoint')
        verified[name] = digest.hexdigest()
    return verified


def check_weight_shapes(stored, expected, tied_embeddings=False):
    """The only permitted omission is the safetensors-deduplicated tied LM head."""
    missing, extra = set(expected) - set(stored), set(stored) - set(expected)
    alias = 'plm.lm_head.weight'
    embedding = 'plm.model.language_model.embed_tokens.weight'
    allowed = {alias} if tied_embeddings and embedding in stored else set()
    if extra or missing - allowed:
        raise ValueError(f'Incomplete/incompatible checkpoint: missing={sorted(missing - allowed)}, extra={sorted(extra)}')
    for key, shape in stored.items():
        if tuple(shape) != tuple(expected[key]):
            raise ValueError(f'{key}: checkpoint shape {shape} differs from model shape {expected[key]}')
    if alias in missing and tuple(expected[alias]) != tuple(stored[embedding]):
        raise ValueError('Tied output and input embedding shapes disagree')
    return sorted(missing)


def load_model(directory, profile, device):
    verify_files(directory, profile)
    # Import only after the source and its runtime files match the pinned hashes.
    spec = importlib.util.spec_from_file_location('causalvlm_checkpoint', Path(directory)/'modeling_causalvlm.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    from safetensors import safe_open
    dtype = torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32
    model = module.CausalVLM(str(directory), device=device, dtype=dtype)
    with safe_open(str(Path(directory)/'model.safetensors'), framework='pt', device='cpu') as handle:
        stored = {key: handle.get_slice(key).get_shape() for key in handle.keys()}
    expected = {'plm.'+key: tuple(value.shape) for key, value in model.plm.state_dict().items()}
    expected.update({'head.'+key: tuple(value.shape) for key, value in model.head.state_dict().items()})
    tied = model.plm.get_input_embeddings().weight.data_ptr() == model.plm.get_output_embeddings().weight.data_ptr()
    check_weight_shapes(stored, expected, tied_embeddings=tied)
    model.eval()
    return module, model


def predict_video(module, model, video, threshold, max_new_tokens):
    if not 0 <= threshold <= 1 or max_new_tokens < 1:
        raise ValueError('Use an edge threshold in [0, 1] and a positive token limit')
    start = time.monotonic()
    frames = module.uniform_sample_frames(str(video), module.NFRAMES)
    events = model.describe(frames, max_new_tokens=max_new_tokens)
    descriptions = [description for _, description in events]
    if descriptions:
        matrix, _ = model.causal_graph(frames, descriptions, threshold)
        matrix = matrix.tolist()
        # Only forward pairs are eligible, including at a zero threshold.
        edges = [(i + 1, j + 1) for i in range(len(events)) for j in range(i + 1, len(events))
                 if matrix[i][j] >= threshold]
    else:
        matrix, edges = [], []
    return {'video': str(Path(video).resolve()), 'events': events,
            'causal_edges': edges, 'adjacency_matrix': matrix,
            'elapsed_seconds': time.monotonic() - start}


def write_result(path, result):
    path = Path(path)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)  # Atomic creation; an existing destination raises FileExistsError.
        finally:
            temporary.unlink(missing_ok=True)


def main():
    profile = json.loads(Path(__file__).with_name('published_checkpoint.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=profile['repo_id'], help='Pinned Hugging Face model ID or its local export directory')
    parser.add_argument('--video', nargs='+', required=True, help='One or more source video files')
    parser.add_argument('--output', required=True, help='New JSON result file')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--edge-threshold', type=float, default=.4, help='Published wrapper default: 0.4')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='Published wrapper default: 256')
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError('Choose a new prediction output path')
    if not 0 <= args.edge_threshold <= 1 or args.max_new_tokens < 1:
        parser.error('Use an edge threshold in [0, 1] and a positive token limit')
    for video in args.video:
        if not Path(video).is_file():
            raise FileNotFoundError(video)
    directory = Path(args.checkpoint)
    if not directory.is_dir():
        if args.checkpoint != profile['repo_id']:
            parser.error('checkpoint must be a local export directory or causalvlm/CausalVLM')
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(repo_id=profile['repo_id'], revision=profile['revision'],
                                          allow_patterns=list(profile['files'])))
    module, model = load_model(directory, profile, args.device)
    predictions = []
    for video in args.video:
        prediction = predict_video(module, model, video, args.edge_threshold, args.max_new_tokens)
        predictions.append(prediction)
        print(f'{Path(video).name}: {len(prediction["events"])} events, {len(prediction["causal_edges"])} edges', flush=True)
    write_result(args.output, {'model_id': profile['repo_id'], 'revision': profile['revision'],
        'edge_threshold': args.edge_threshold,
        'max_new_tokens': args.max_new_tokens, 'predictions': predictions})


if __name__ == '__main__':
    main()
