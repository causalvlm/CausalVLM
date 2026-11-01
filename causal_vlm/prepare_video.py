"""Create 64 scene-aware frames with source-video timestamps for training/inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .src.data.sampling import scene_indices


@torch.no_grad()
def features_and_times(filename, model, processor, device, batch_size):
    import av
    embeddings, times, batch = [], [], []
    def flush():
        if batch:
            pixels = processor(images=batch, return_tensors='pt')['pixel_values'].to(device)
            embeddings.append(model(pixel_values=pixels).last_hidden_state[:, 0].float().cpu().numpy())
            batch.clear()
    with av.open(str(filename)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        origin = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Decoded frames require source timestamps')
            time = float(frame.pts * frame.time_base)
            if origin is None:
                origin = time
            times.append(time-origin)
            if len(times) > 1 and times[-1] < times[-2]:
                raise ValueError('Video timestamps are not monotonic')
            # Resize before retaining a batch of decoded images to bound host RAM.
            batch.append(frame.to_image().resize((518, 518)))
            if len(batch) == batch_size:
                flush()
        flush()
    if not times:
        raise ValueError('Video contains no decoded frames')
    interval = 1/rate if rate > 0 else (times[-1]-times[-2] if len(times) > 1 else 0)
    duration = max(duration, times[-1]+interval)
    if duration <= times[-1]:
        raise ValueError('Cannot establish a video duration beyond its final frame')
    return np.concatenate(embeddings), times, duration


def save_selected(filename, indices, directory):
    import av
    directory.mkdir(parents=True, exist_ok=False)
    wanted = set(indices)
    with av.open(str(filename)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                frame.to_image().save(directory/f'{index:08d}.jpg', quality=95)
                wanted.remove(index)
            if not wanted:
                break
    if wanted:
        raise ValueError('Video changed or could not be decoded on the second pass')
    return [f'{index:08d}.jpg' for index in indices]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos', required=True, help='JSON mapping video ID to source video path')
    p.add_argument('--output-dir', required=True, help='New directory, e.g. data/frames_train')
    p.add_argument('--annotations', help='DCC JSON; omit when preparing inference inputs')
    p.add_argument('--split', default='train')
    p.add_argument('--dino-checkpoint', default='facebook/dinov2-small')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch-size must be positive')
    from transformers import AutoImageProcessor, AutoModel
    videos_path = Path(a.videos)
    videos = json.loads(videos_path.read_text())
    annotations = json.loads(Path(a.annotations).read_text()) if a.annotations else None
    if annotations is not None and annotations.keys() != videos.keys():
        raise ValueError('Video manifest and annotations must contain exactly the same IDs')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    model = AutoModel.from_pretrained(a.dino_checkpoint).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(a.dino_checkpoint)
    inference, training = {}, []
    for vid, filename in videos.items():
        if not vid or Path(vid).name != vid or vid in {'.', '..'}:
            raise ValueError('Video IDs must be safe single directory names')
        filename = videos_path.parent / filename
        features, timestamps, duration = features_and_times(filename, model, processor, device, a.batch_size)
        selected = scene_indices(features, 64, 0.8)
        names = save_selected(filename, selected, out/'frames'/vid)
        paths = [f'{vid}/{name}' for name in names]
        times = [timestamps[index] for index in selected]
        metadata = {'duration': duration, 'timestamps': times, 'frame_paths': paths,
                    'frame_indices': selected, 'dino_checkpoint': a.dino_checkpoint,
                    'scene_threshold': 0.8, 'representative': 'mid-frame'}
        (out/'metadata').mkdir(exist_ok=True)
        (out/'metadata'/f'{vid}.json').write_text(json.dumps(metadata, indent=2))
        inference[vid] = {'duration': duration, 'frame_timestamps': times,
                          'frame_paths': [f'frames/{path}' for path in paths]}
        if annotations is not None:
            if not isinstance(annotations[vid], dict):
                raise ValueError(f'{vid}: DCC annotation is missing or invalid')
            training.append({'video_id': vid, 'duration': duration, 'frame_paths': paths,
                             'frame_timestamps': times, 'original_events': annotations[vid]['events']})
        print(f'Prepared {vid}', flush=True)
    (out/'inference_inputs.json').write_text(json.dumps(inference, indent=2))
    if annotations is not None:
        (out/'annotations').mkdir()
        (out/'annotations'/f'{a.split}_processed.json').write_text(json.dumps(training, indent=2))


if __name__ == '__main__':
    main()
