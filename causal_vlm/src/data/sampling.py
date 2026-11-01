"""DINOv2 scene balancing used for training and inference preprocessing."""
from __future__ import annotations
import numpy as np


def scene_indices(features, count=64, threshold=0.8):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or count < 1:
        raise ValueError('Expected nonempty finite frame embeddings and a positive sample count')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any() or not -1 <= threshold <= 1:
        raise ValueError('Embedding norms must be positive and cosine threshold in [-1, 1]')
    x = x / norms
    similarity = (x[:-1] * x[1:]).sum(1)
    cuts = [0] + (np.flatnonzero(similarity < threshold)+1).tolist() + [len(x)]
    scenes = list(zip(cuts[:-1], cuts[1:]))
    target = min(count, len(x))
    while len(scenes) > target:
        midpoints = [(a+b-1)//2 for a, b in scenes]
        scores = [float(x[i] @ x[j]) for i, j in zip(midpoints[:-1], midpoints[1:])]
        merge = int(np.argmax(scores))
        scenes[merge:merge+2] = [(scenes[merge][0], scenes[merge+1][1])]
    while len(scenes) < target:
        index = max(range(len(scenes)), key=lambda i: scenes[i][1]-scenes[i][0])
        start, end = scenes[index]
        cut = start + int(np.argmin(similarity[start:end-1])) + 1
        scenes[index:index+1] = [(start, cut), (cut, end)]
    # One mid-frame per balanced scene; ties resolve to the earliest frame.
    selected = [(a+b-1)//2 for a, b in scenes]
    if len(selected) < count:
        selected = np.asarray(selected)[np.linspace(0, len(selected)-1, count).round().astype(int)].tolist()
    return selected
