"""Load preprocessed DCC videos without changing their annotation graphs."""
from __future__ import annotations

import ast
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..events import format_events


def annotation_events(record):
    events = record.get("original_events")
    if events is None:
        events = record.get("events", [])
    if isinstance(events, str):
        events = ast.literal_eval(events)
    if not isinstance(events, list):
        raise ValueError("Events must be a list")
    cleaned = []
    for i, event in enumerate(events):
        timestamp = event.get("timestamp", event.get("timestamps"))
        if not timestamp or len(timestamp) != 2:
            raise ValueError("Every annotation needs an explicit start and end")
        start, end = map(float, timestamp)
        if not all(map(math.isfinite, (start, end))) or not 0 <= start < end:
            raise ValueError("Invalid annotation interval")
        cleaned.append({**event, "event_id": event.get("event_id", i+1),
                        "timestamp": [start, end], "description": event.get("description", "")})
    return sorted(cleaned, key=lambda e: e["timestamp"][0])


def annotation_adjacency(events):
    ids = {e["event_id"]: i for i, e in enumerate(events)}
    if len(ids) != len(events):
        raise ValueError("Duplicate event IDs")
    adjacency = torch.zeros(len(events), len(events))
    for i, event in enumerate(events):
        chain = event.get("chain") or {}
        edges = [(event["event_id"], dst) for dst in chain.get("effect_event", [])]
        edges += [(src, event["event_id"]) for src in chain.get("cause_event", [])]
        for src, dst in edges:
            if src not in ids or dst not in ids:
                raise ValueError("Causal edge references an unknown event ID")
            if ids[src] >= ids[dst]:
                raise ValueError("Causal edges must follow event onset order")
            adjacency[ids[src], ids[dst]] = 1
    return adjacency


def flatten_paths(paths):
    return [path for group in paths for path in ([group] if isinstance(group, str) else group)]


class SimplePreprocessedCausalVideoDataset(Dataset):
    def __init__(self, data_path, split="train", resolution=448, augment=True,
                 cache_size=0, num_frames=64):
        self.base_path = Path(data_path)
        self.frames_dir = self.base_path / f"frames_{split}" / "frames"
        self.meta_dir = self.base_path / f"frames_{split}" / "metadata"
        self.resolution = resolution
        self.augment = augment and split == "train"
        self.cache_size = cache_size
        self.cache = {}
        self.num_frames = num_frames
        with (self.base_path / f"frames_{split}/annotations/{split}_processed.json").open() as f:
            self.samples = json.load(f)
        if not isinstance(self.samples, list):
            raise ValueError("Preprocessed annotations must be a list of video records")
        ids = [r["video_id"] for r in self.samples]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate video IDs in preprocessed annotations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Include the video ID when reporting sample-loading failures.
        record = self.samples[idx]
        try:
            return self._load(record)
        except (ValueError, KeyError, OSError) as exc:
            raise ValueError(f"Video {record.get('video_id', idx)}: {exc}") from exc

    def _load(self, record):
        paths = flatten_paths(record.get("frame_paths", []))
        if len(paths) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, received {len(paths)}")
        metadata_path = self.meta_dir / f"{record['video_id']}.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        timestamps = record.get("frame_timestamps")
        if timestamps is None:
            timestamps = metadata.get("timestamps")
            meta_paths = flatten_paths(metadata.get("frame_paths", []))
            if meta_paths and timestamps is not None:
                if len(meta_paths) != len(timestamps) or len(set(meta_paths)) != len(meta_paths):
                    raise ValueError("Frame metadata paths and timestamps are inconsistent")
                by_path = dict(zip(meta_paths, timestamps))
                timestamps = [by_path[p] for p in paths]
        if timestamps is None or len(timestamps) != len(paths):
            raise ValueError("Sampled frame timestamps are required; regenerate preprocessing metadata")
        frame_ts = torch.tensor(timestamps, dtype=torch.float32)
        if not torch.isfinite(frame_ts).all() or (frame_ts < 0).any() or (frame_ts[1:] < frame_ts[:-1]).any():
            raise ValueError("Frame timestamps must be finite, nonnegative, and ordered")
        duration = record.get("duration", metadata.get("duration"))
        if duration is None and metadata.get("num_frames") and metadata.get("fps"):
            duration = metadata["num_frames"] / metadata["fps"]
        if duration is None or not math.isfinite(float(duration)) or float(duration) <= 0:
            raise ValueError("Actual video duration is required (not the last annotation end)")
        duration = float(duration)
        if (frame_ts > duration).any():
            raise ValueError("Frame timestamp exceeds video duration")
        events = annotation_events(record)
        if any(e["timestamp"][1] > duration for e in events):
            raise ValueError("Annotation end exceeds video duration")
        frames = np.stack([self._load_frame(self.frames_dir / p) for p in paths])
        if self.augment:
            if random.random() > 0.5:
                frames = frames[:, :, ::-1, :].copy()
            if random.random() > 0.5:
                frames = np.clip(frames * random.uniform(0.8, 1.2), 0, 255)
        frames = (frames.astype(np.float32) / 255).transpose(0, 3, 1, 2)
        return {"frames": torch.from_numpy(frames).unsqueeze(1),
                "frame_timestamps": frame_ts,
                "event_timestamps": torch.tensor([e["timestamp"] for e in events], dtype=torch.float32).reshape(-1, 2),
                "adjacency_matrix": annotation_adjacency(events),
                "caption": format_events(events), "events": events,
                "event_descriptions": [e["description"] for e in events],
                "num_events": len(events), "video_id": record["video_id"], "duration": duration}

    def _load_frame(self, path):
        key = str(path)
        if key in self.cache:
            return self.cache[key].copy()
        with Image.open(path) as image:
            array = np.array(image.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.BILINEAR))
        if len(self.cache) < self.cache_size:
            self.cache[key] = array.copy()
        return array


def simple_collate_fn(batch):
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    n_max = max(x["num_events"] for x in batch)
    event_ts = torch.zeros(len(batch), n_max, 2)
    adjacency = torch.zeros(len(batch), n_max, n_max)
    event_masks = torch.zeros(len(batch), n_max, dtype=torch.bool)
    for b, sample in enumerate(batch):
        n = sample["num_events"]
        event_ts[b, :n] = sample["event_timestamps"]
        adjacency[b, :n, :n] = sample["adjacency_matrix"]
        event_masks[b, :n] = True
    return {"frames": torch.stack([x["frames"] for x in batch]),
            "frame_timestamps": torch.stack([x["frame_timestamps"] for x in batch]),
            "event_timestamps": event_ts, "adjacency_matrix": adjacency, "event_masks": event_masks,
            "captions": [x["caption"] for x in batch], "events": [x["events"] for x in batch],
            "event_descriptions": [x["event_descriptions"] for x in batch],
            "num_events": [x["num_events"] for x in batch],
            "video_ids": [x["video_id"] for x in batch], "durations": [x["duration"] for x in batch]}
