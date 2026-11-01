"""Prepare proposed-pair clips and blank human-rating templates for DCC construction."""

import argparse
import csv
import hashlib
import math
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from media import find_video as _find_video
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from schema import validate_annotations, write_json, write_text

# Default video sample sizes and event-count stratum.

ACTIVITYNET_SAMPLE   = 1_200
YOUCOOK2_SAMPLE      = 800
HIGH_EVENT_THRESHOLD = 5      # videos with >= this many events are oversampled
RANDOM_SEED          = 42

# Video utilities



def _extract_segment(video_path: str, start: float, end: float, out_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start), "-to", str(end),
        "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
        out_path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _concat_segments(segment_paths: List[str], out_path: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for seg in segment_paths:
            f.write(f"file '{os.path.abspath(seg)}'\n")
        concat_list = f.name
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c", "copy", out_path,
    ]
    ok = subprocess.run(cmd, capture_output=True).returncode == 0
    os.unlink(concat_list)
    return ok


def _build_merged_clip(
    video_path: str,
    cause_event: Dict[str, Any],
    effect_event: Dict[str, Any],
    out_path: str,
) -> bool:
    """
    Extract cause and effect segments and concatenate them in temporal order.

    Returns True if the merged clip was written successfully.
    """
    ordered = sorted(
        [cause_event, effect_event],
        key=lambda e: e["timestamp"][0],
    )
    with tempfile.TemporaryDirectory() as tmp:
        segments = []
        for i, event in enumerate(ordered):
            start, end = event["timestamp"]
            seg = os.path.join(tmp, f"seg_{i}.mp4")
            if not _extract_segment(video_path, start, end, seg):
                return False
            segments.append(seg)

        if len(segments) == 1:
            import shutil
            shutil.copy2(segments[0], out_path)
            return True
        return _concat_segments(segments, out_path)


# Annotation edge extraction

def _extract_edges(events: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Return all directed causal edges (cause_id, effect_id)."""
    edges = []
    for event in events:
        src = event["event_id"]
        for dst in event["chain"].get("effect_event", []):
            edges.append((src, int(dst)))
    return edges


# Stratified sampling

def _stratified_sample(video_event_counts, n_sample, rng, high_weight,
                       threshold=HIGH_EVENT_THRESHOLD):
    """Allocate by population size times an explicit high-stratum weight > 1.

    Round the high quota upward, cap it at availability, and fill the remainder
    from the low stratum. This oversamples high-event videos whenever the
    population and sample budget allow it, including high-dominated populations.
    """
    if type(n_sample) is not int or not 0 < n_sample <= len(video_event_counts):
        raise ValueError('Requested sample must be positive and no larger than the eligible population')
    if not math.isfinite(high_weight) or high_weight <= 1:
        raise ValueError('high-event-weight must be finite and greater than 1')
    low = sorted(v for v, n in video_event_counts.items() if n < threshold)
    high = sorted(v for v, n in video_event_counts.items() if n >= threshold)
    rng.shuffle(low)
    rng.shuffle(high)
    fraction = len(high) / (len(high) + len(low) / high_weight)
    n_high = min(len(high), max(n_sample - len(low), math.ceil(n_sample * fraction)))
    selected = high[:n_high] + low[:n_sample - n_high]
    rng.shuffle(selected)
    return selected


def _selection(annotations, dataset, n_sample, rng, high_weight):
    counts = {vid: len(record['events']) for vid, record in annotations.items()
              if _extract_edges(record['events'])}
    selected = _stratified_sample(counts, n_sample, rng, high_weight)
    high_available = sum(n >= HIGH_EVENT_THRESHOLD for n in counts.values())
    high_selected = sum(counts[v] >= HIGH_EVENT_THRESHOLD for v in selected)
    return {'dataset': dataset, 'requested_videos': n_sample, 'selected_video_ids': selected,
            'eligible_videos': len(counts), 'selected_videos': len(selected),
            'eligible_high_event_videos': high_available,
            'selected_high_event_videos': high_selected,
            'selected_low_event_videos': len(selected) - high_selected}


def _generate_tasks_for_dataset(annotations, video_dir, dataset_name, sampled_ids, clips_dir):
    """Create one record for every selected pair, including failed clips."""
    tasks = []
    clips_dir.mkdir(parents=True, exist_ok=True)
    for video_id in tqdm(sampled_ids, desc=f'Building clips [{dataset_name}]'):
        events = annotations[video_id]['events']
        event_map = {e['event_id']: e for e in events}
        video_path = _find_video(video_id, video_dir)
        for src_id, dst_id in _extract_edges(events):
            cause_ev, effect_ev = event_map[src_id], event_map[dst_id]
            pair_id = f'{dataset_name}_{video_id}_{src_id}_{dst_id}'
            clip_path = clips_dir / f'{pair_id}.mp4'
            clip_ok, error = False, None
            if video_path is None:
                error = 'source_video_missing'
            else:
                try:
                    clip_ok = _build_merged_clip(video_path, cause_ev, effect_ev, str(clip_path))
                    clip_ok = bool(clip_ok and clip_path.is_file() and clip_path.stat().st_size > 0)
                    if not clip_ok:
                        error = 'clip_extraction_failed'
                except (OSError, subprocess.SubprocessError) as exc:
                    error = f'clip_extraction_failed: {exc}'
            tasks.append({
                'pair_id': pair_id, 'dataset': dataset_name, 'video_id': video_id,
                'cause_event_id': src_id, 'effect_event_id': dst_id,
                'cause_description': cause_ev['description'], 'effect_description': effect_ev['description'],
                'cause_timestamp': cause_ev['timestamp'], 'effect_timestamp': effect_ev['timestamp'],
                'n_events_in_video': len(events), 'high_event_stratum': len(events) >= HIGH_EVENT_THRESHOLD,
                'clip_path': str(clip_path) if clip_ok else None, 'clip_ok': clip_ok,
                'error': error, 'rater_1': None, 'rater_2': None, 'rater_3': None})
    return tasks


def generate_tasks(actnet_path, yc2_path, actnet_video_dir, yc2_video_dir, output_dir,
                   *, high_weight, actnet_sample=ACTIVITYNET_SAMPLE,
                   yc2_sample=YOUCOOK2_SAMPLE, seed=RANDOM_SEED):
    """Publish ready tasks only when all selected pairs have usable clips."""
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new task directory; existing ratings are protected: {out}')
    rng = random.Random(seed)
    datasets = [('activitynet', actnet_path, actnet_video_dir, actnet_sample),
                ('youcook2', yc2_path, yc2_video_dir, yc2_sample)]
    prepared, selections = [], {}
    for name, path, video_dir, count in datasets:
        data = validate_annotations(json.loads(Path(path).read_text()))
        selection = _selection(data, name, count, rng, high_weight)
        selection['input_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        selections[name] = selection
        prepared.append((name, data, video_dir, selection['selected_video_ids']))
    out.mkdir(parents=True, exist_ok=False)
    # Record selected IDs before clip generation for traceability.
    write_json(out / 'selection.json', {'seed': seed, 'high_event_threshold': HIGH_EVENT_THRESHOLD,
        'high_event_weight': high_weight, 'allocation': 'ceil(n * wH / (wH + L)), capped by availability',
        'datasets': selections})
    tasks = []
    for name, data, video_dir, selected in prepared:
        tasks.extend(_generate_tasks_for_dataset(data, video_dir, name, selected, out / 'clips'))
    rng.shuffle(tasks)
    failed = sum(not t['clip_ok'] for t in tasks)
    write_json(out / 'sampling_stats.json', {'complete': failed == 0,
        'total_tasks': len(tasks), 'clips_created': len(tasks) - failed, 'clips_failed': failed,
        'datasets': selections})
    if failed:
        write_json(out / 'tasks.partial.json', tasks)
        raise RuntimeError(f'{failed} selected pairs lack usable clips. See tasks.partial.json; no ready tasks or ratings template were written.')
    write_json(out / 'tasks.json', tasks)
    with (out / 'annotations_template.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['pair_id', 'rater_1', 'rater_2', 'rater_3'])
        writer.writerows([t['pair_id'], '', '', ''] for t in tasks)
    write_text(out / 'INSTRUCTIONS.txt',
        'View each merged cause/effect clip and its two captions. Judge whether, in this video, '
        'the initiating event enables the later event as an observed prerequisite. Temporal succession '
        'alone is insufficient; do not infer universal or interventional necessity.\n'
        'Each of three annotators independently supplies one integer rating: '
        '1 definitely not causal; 2 probably not causal; 3 uncertain/borderline; '
        '4 probably causal; 5 definitely causal. Leave no rating blank.\n'
        'The retention cutoff is an explicit input to dcc_apply_human_ratings.py.\n')
    print(f'Created {len(tasks)} complete pair tasks in {out}', flush=True)


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DCC benchmark -- human validation task generation (Step 4a of 4)"
        )
    )
    parser.add_argument(
        "--activitynet", required=True, metavar="FILE",
        help="Step-3 VLM-validated annotation JSON for ActivityNet",
    )
    parser.add_argument(
        "--youcook2", required=True, metavar="FILE",
        help="Step-3 VLM-validated annotation JSON for YouCook2",
    )
    parser.add_argument(
        "--actnet-video-dir", required=True, metavar="DIR",
        help="Root directory containing ActivityNet video files",
    )
    parser.add_argument(
        "--yc2-video-dir", required=True, metavar="DIR",
        help="Root directory containing YouCook2 video files",
    )
    parser.add_argument(
        "--output-dir", default="human_study", metavar="DIR",
        help="Directory for tasks.json, clips/, and annotation template (default: human_study/)",
    )
    parser.add_argument('--high-event-weight', type=float, required=True,
                        help='Relative allocation weight > 1 for videos with at least five events')
    parser.add_argument('--activitynet-sample', type=int, default=ACTIVITYNET_SAMPLE)
    parser.add_argument('--youcook2-sample', type=int, default=YOUCOOK2_SAMPLE)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    generate_tasks(
        actnet_path      = args.activitynet,
        yc2_path         = args.youcook2,
        actnet_video_dir = args.actnet_video_dir,
        yc2_video_dir    = args.yc2_video_dir,
        output_dir       = args.output_dir,
        high_weight      = args.high_event_weight,
        actnet_sample    = args.activitynet_sample,
        yc2_sample       = args.youcook2_sample,
        seed             = args.seed,
    )


if __name__ == "__main__":
    main()
