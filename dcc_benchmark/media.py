"""Resolve source videos using the IDs present in the released DCC labels."""
from glob import escape
from pathlib import Path


def find_video(video_id, video_dir):
    base = Path(video_dir)
    stems = [video_id, video_id[2:] if video_id.startswith('v_') else f'v_{video_id}']
    extensions = ('.mp4', '.mkv', '.webm', '.avi')
    for stem in stems:
        for extension in extensions:
            candidate = base / (stem + extension)
            if candidate.is_file():
                return str(candidate)
    for stem in stems:
        matches = sorted(p for p in base.rglob(escape(stem) + '.*')
                         if p.is_file() and p.suffix.lower() in extensions)
        if len(matches) > 1:
            raise ValueError(f'Ambiguous source video for {video_id}: {matches}')
        if matches:
            return str(matches[0])
    return None
