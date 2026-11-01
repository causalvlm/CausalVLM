"""Timestamped event serialization shared by data loading and inference."""
from __future__ import annotations

import math
import re

NUMBER = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
EXPLICIT = re.compile(rf"^\s*({NUMBER})s\s+({NUMBER})s\s+(.+)$", re.S)
ONSET = re.compile(rf"^\s*({NUMBER})s\s+(.+)$", re.S)


def format_events(events):
    return " ".join(f"<event> {e['timestamp'][0]:.2f}s {e['timestamp'][1]:.2f}s {e['description']} </event>"
                    for e in events)


def parse_events(text, duration):
    """Parse explicit intervals and onset-only captions.

    Onset-only interval ends use the next strictly later onset or the video duration.
    Explicit intervals preserve overlaps. Malformed events are omitted; an empty
    prediction remains a valid empty graph, retaining the generated text for inspection.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The actual video duration must be finite and positive")
    candidates = []
    for body in re.findall(r"<event>(.*?)</event>", text, re.S):
        explicit = EXPLICIT.match(body)
        onset = None if explicit else ONSET.match(body)
        if explicit:
            start, end, desc = float(explicit[1]), float(explicit[2]), explicit[3].strip()
        elif onset:
            start, end, desc = float(onset[1]), None, onset[2].strip()
        else:
            continue
        if not desc or not math.isfinite(start) or not 0 <= start < duration:
            continue
        if end is not None and (not math.isfinite(end) or not start < end <= duration):
            continue
        candidates.append((start, end, desc))
    candidates.sort(key=lambda ev: ev[0])
    events = []
    for i, (start, end, desc) in enumerate(candidates):
        if end is None:
            end = next((other[0] for other in candidates[i+1:] if other[0] > start), duration)
        events.append({"event_id": i + 1, "timestamp": [start, end], "description": desc})
    return events
