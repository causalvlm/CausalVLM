import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DINOV2_CACHE: dict = {}


def _get_dinov2(device: torch.device):
    worker = (torch.utils.data.get_worker_info() or type("w", (), {"id": 0})()).id
    if worker not in DINOV2_CACHE:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.to(device).eval()
        DINOV2_CACHE[worker] = model
    return DINOV2_CACHE[worker]


class SimplePreprocessedCausalVideoDataset(Dataset):
    def __init__(
        self,
        data_path:  str,
        split:      str  = "train",
        resolution: int  = 224,
        augment:    bool = True,
        cache_size: int  = 0,
    ) -> None:
        self.base_path  = Path(data_path)
        self.frames_dir = self.base_path / f"frames_{split}" / "frames"
        self.resolution = resolution
        self.augment    = augment and split == "train"
        self.cache      = {} if cache_size > 0 else None
        self.cache_size = cache_size

        ann_file = self.base_path / f"frames_{split}" / "annotations" / f"{split}_processed.json"
        try:
            with open(ann_file) as f:
                self.samples = json.load(f)
        except FileNotFoundError:
            print(f"Annotation file not found: {ann_file}")
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self._get_with_retry(idx)

    def _get_with_retry(self, idx: int, max_retries: int = 100) -> dict:
        if not self.samples:
            raise IndexError("Dataset is empty.")
        for attempt in range(max_retries):
            try:
                item = self._load((idx + attempt) % len(self.samples))
                if item is not None:
                    return item
            except Exception:
                continue
        raise RuntimeError(f"No valid sample found after {max_retries} attempts.")

    def _load(self, idx: int) -> dict | None:
        ann             = self.samples[idx]
        clip_paths_list = ann.get("frame_paths", [])

        if len(clip_paths_list) < 8:
            return None
        clip_paths_list = clip_paths_list[:8]

        clips = []
        for clip_frame_paths in clip_paths_list:
            if len(clip_frame_paths) != 8:
                return None
            clip = self._load_clip(clip_frame_paths)
            if clip is None or len(clip) != 8:
                return None
            clips.append(clip)

        if len(clips) != 8:
            return None

        try:
            stacked = np.stack(clips)
        except ValueError:
            return None

        if self.augment:
            stacked = self._augment(stacked)
        frames_tensor = self._normalise(stacked)

        events = ann.get("original_events") or ann.get("events", [])
        if not events:
            return None
        try:
            events.sort(key=lambda e: e["timestamp"][0])
        except (IndexError, KeyError, TypeError):
            return None

        caption_parts = []
        event_timestamps = []
        fps = ann.get("fps", 30.0)
        event_boundaries = []
        event_descriptions = []

        for e in events:
            try:
                start, end  = e["timestamp"][0], e["timestamp"][1]
                desc        = e.get("description", "")
                caption_parts.append(f"<event> {start:.2f}s {desc} </event>")
                event_timestamps.append([start, end])
                event_boundaries.append([int(start * fps), int(end * fps)])
                event_descriptions.append(desc)
            except (IndexError, KeyError, TypeError, ValueError):
                continue

        if not caption_parts:
            return None

        n_events = len(events)
        adj = np.zeros((n_events, n_events), dtype=np.float32)
        id2idx = {e.get("event_id", i): i for i, e in enumerate(events)}

        for i, e in enumerate(events):
            for eid in e.get("chain", {}).get("effect_event", []):
                j = id2idx.get(eid)
                if j is not None and 0 <= i < n_events and 0 <= j < n_events:
                    adj[i, j] = 1.0

        if adj.sum() == 0 and n_events > 1:
            for i in range(n_events - 1):
                adj[i, i + 1] = 1.0

        return {
            "frames":             torch.from_numpy(frames_tensor).float(),
            "event_boundaries":   event_boundaries,
            "event_timestamps":   torch.tensor(event_timestamps).float(),
            "adjacency_matrix":   torch.from_numpy(adj).float(),
            "caption":            "".join(caption_parts),
            "video_id":           ann["video_id"],
            "event_descriptions": event_descriptions,
            "num_events":         n_events,
        }

    def _load_frame(self, path: Path) -> np.ndarray | None:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize(
                    (self.resolution, self.resolution), Image.BILINEAR
                )
                return np.array(img)
        except Exception:
            pass
        try:
            f = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if f is not None:
                return cv2.resize(
                    cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                    (self.resolution, self.resolution),
                )
        except Exception:
            pass
        return None

    def _load_clip(self, frame_paths: list) -> np.ndarray | None:
        frames, last = [], None
        for p in frame_paths:
            full = self.frames_dir / p
            if self.cache and str(full) in self.cache:
                f = self.cache[str(full)].copy()
            else:
                f = self._load_frame(full)
                if self.cache is not None and f is not None and len(self.cache) < self.cache_size:
                    self.cache[str(full)] = f.copy()
            if f is None:
                f = last if last is not None else np.zeros(
                    (self.resolution, self.resolution, 3), dtype=np.uint8
                )
            else:
                last = f
            frames.append(f)
        return np.array(frames) if frames else None

    def _augment(self, clips: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            clips = clips[:, :, :, ::-1, :].copy()
        if random.random() > 0.5:
            clips = np.clip(clips * random.uniform(0.8, 1.2), 0, 255)
        return clips

    def _normalise(self, frames: np.ndarray) -> np.ndarray:
        return (frames.astype(np.float32) / 255.0).transpose(0, 1, 4, 2, 3)


def simple_collate_fn(batch: list[dict]) -> dict[str, Any]:
    frames    = torch.stack([x["frames"] for x in batch])
    captions  = [x["caption"] for x in batch]
    video_ids = [x["video_id"] for x in batch]

    # Event timestamps — pad to max N in batch
    ts_list = [x.get("event_timestamps") for x in batch]
    if ts_list and all(t is not None for t in ts_list):
        max_n = max(t.shape[0] for t in ts_list)
        padded_ts, masks = [], []
        for t in ts_list:
            n = t.shape[0]
            pad = torch.zeros(max_n - n, 2)
            padded_ts.append(torch.cat([t, pad]))
            masks.append(torch.cat([torch.ones(n, dtype=torch.bool),
                                    torch.zeros(max_n - n, dtype=torch.bool)]))
        event_timestamps = torch.stack(padded_ts)
        event_masks      = torch.stack(masks)
    else:
        event_timestamps = event_masks = None

    # Adjacency matrices — pad to max N
    adj_list = [x.get("adjacency_matrix") for x in batch]
    if adj_list and all(a is not None for a in adj_list):
        max_n = max(a.shape[0] for a in adj_list)
        padded = []
        for a in adj_list:
            n   = a.shape[0]
            pad = torch.zeros(max_n, max_n)
            pad[:n, :n] = a
            padded.append(pad)
        adjacency_matrix = torch.stack(padded)
    else:
        adjacency_matrix = None

    return {
        "frames":             frames,
        "captions":           captions,
        "video_ids":          video_ids,
        "event_timestamps":   event_timestamps,
        "event_masks":        event_masks,
        "adjacency_matrix":   adjacency_matrix,
        "event_boundaries":   [x.get("event_boundaries", []) for x in batch],
        "event_descriptions": [x.get("event_descriptions", []) for x in batch],
        "num_events":         [x.get("num_events", 0) for x in batch],
    }
