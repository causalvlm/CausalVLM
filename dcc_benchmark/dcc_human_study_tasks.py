"""
DCC Benchmark -- Step 4a: Human Validation Task Generation
===========================================================
Generates the annotation tasks for the Tier-3 human validation of DCC causal
pair annotations.

Implements the sampling strategy described in Section 3.2 of the paper:

    "Following [chen2011collecting, xu2016msr, nadeem2025narrativebridge], we
     sample 2,000 videos (1,200 ActivityNet, 800 YouCook2) containing 9,574
     causal pairs, stratified to oversample videos with 5+ events where
     annotation risk is highest.  Three annotators view merged clips and rate
     causality (1=definitely not, 5=definitely yes)."

For each sampled causal pair the script:
  1. Extracts the cause and effect video segments using ffmpeg.
  2. Concatenates them into a single merged clip that annotators view.
  3. Writes a task manifest (tasks.json) with metadata for every pair.

The task manifest is the bridge between this script and the scoring script
(dcc_human_study_score.py).  Annotators fill in their ratings and the scoring
script reads the completed CSV back alongside tasks.json.

Pipeline
--------
    Step 1 -- dcc_annotation_generator.py
    Step 2 -- dcc_emscore_validation.py
    Step 3 -- dcc_vlm_validation.py
    Step 4a -- dcc_human_study_tasks.py      (this file)
    Step 4b -- dcc_human_study_score.py

Sampling parameters (Section 3.2)
----------------------------------
    ActivityNet sample : 1,200 videos
    YouCook2 sample    : 800 videos
    High-event stratum : videos with >= 5 events (oversampled at 60 % of budget)
    Random seed        : 42 (for reproducibility)

Annotation rating scale
-----------------------
    1 = definitely not causal
    2 = probably not causal
    3 = uncertain / borderline
    4 = probably causal
    5 = definitely causal

    A pair is retained if its mean rating across three annotators is >= 3.0.

Requirements
------------
    ffmpeg >= 4.0 (must be on PATH)

Usage
-----
    python dcc_human_study_tasks.py \\
        --activitynet dcc_actnet_vlm.json \\
        --youcook2    dcc_yc2_vlm.json \\
        --actnet-video-dir  /data/activitynet/videos \\
        --yc2-video-dir     /data/youcook2/videos \\
        --output-dir        human_study/
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sampling constants (Section 3.2 of the paper)
# ---------------------------------------------------------------------------

ACTIVITYNET_SAMPLE   = 1_200
YOUCOOK2_SAMPLE      = 800
HIGH_EVENT_THRESHOLD = 5      # videos with >= this many events are oversampled
HIGH_STRATUM_FRAC    = 0.60   # fraction of each budget from high-event stratum
RANDOM_SEED          = 42

# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def _find_video(video_id: str, video_dir: str) -> Optional[str]:
    """
    Locate the raw video file for *video_id* under *video_dir*.

    Handles both flat layouts (ActivityNet) and nested recipe subdirectories
    (YouCook2) via recursive glob fallback.
    """
    base = Path(video_dir)
    for candidate in [
        base / f"{video_id}.mp4",
        base / f"v_{video_id}.mp4",
        base / f"{video_id}.mkv",
        base / f"{video_id}.webm",
    ]:
        if candidate.exists():
            return str(candidate)
    matches = sorted(base.rglob(f"{video_id}.*"))
    return str(matches[0]) if matches else None


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


# ---------------------------------------------------------------------------
# Annotation edge extraction
# ---------------------------------------------------------------------------

def _extract_edges(events: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Return all directed causal edges (cause_id, effect_id)."""
    edges = []
    for event in events:
        src = event["event_id"]
        for dst in event["chain"].get("effect_event", []):
            edges.append((src, int(dst)))
    return edges


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _stratified_sample(
    video_event_counts: Dict[str, int],
    n_sample: int,
    rng: random.Random,
    threshold: int = HIGH_EVENT_THRESHOLD,
    high_frac: float = HIGH_STRATUM_FRAC,
) -> List[str]:
    """
    Return a stratified sample of *n_sample* video IDs from *video_event_counts*,
    oversampling videos with >= *threshold* events.

    Allocation:
        high stratum (>= threshold events): floor(n_sample * high_frac) videos
        low  stratum (<  threshold events): remainder

    If either stratum has fewer videos than its allocation, the shortfall is
    filled from the other stratum.

    Parameters
    ----------
    video_event_counts : dict mapping video_id -> number of events in that video
    n_sample           : total number of videos to select
    rng                : seeded Random instance for reproducibility
    """
    low  = [v for v, n in video_event_counts.items() if n < threshold]
    high = [v for v, n in video_event_counts.items() if n >= threshold]

    rng.shuffle(low)
    rng.shuffle(high)

    n_high = min(int(n_sample * high_frac), len(high))
    n_low  = min(n_sample - n_high, len(low))

    # Fill shortfall from the other stratum
    if n_high + n_low < n_sample:
        deficit = n_sample - n_high - n_low
        if len(high) > n_high:
            extra = min(deficit, len(high) - n_high)
            n_high += extra
            deficit -= extra
        if len(low) > n_low and deficit > 0:
            n_low += min(deficit, len(low) - n_low)

    selected = high[:n_high] + low[:n_low]
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def _generate_tasks_for_dataset(
    annotations: Dict[str, Any],
    video_dir: str,
    dataset_name: str,
    n_sample: int,
    clips_dir: Path,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Sample *n_sample* videos from *annotations*, build merged clips for every
    causal pair in the sample, and return a list of task dicts.

    Parameters
    ----------
    annotations  : VLM-validated DCC annotation dict (output of Step 3)
    video_dir    : root directory for raw video files
    dataset_name : "activitynet" or "youcook2" (for task ID prefixes and reports)
    n_sample     : number of videos to sample
    clips_dir    : directory where merged clips are written
    rng          : seeded Random instance
    """
    # Build event-count index for stratified sampling
    video_event_counts: Dict[str, int] = {}
    for video_id, video_data in annotations.items():
        if video_data is None:
            continue
        n_events = len(video_data.get("events", []))
        if n_events > 0 and _extract_edges(video_data["events"]):
            video_event_counts[video_id] = n_events

    available = len(video_event_counts)
    actual_sample = min(n_sample, available)
    if actual_sample < n_sample:
        print(
            f"  [{dataset_name}] Warning: only {available} videos with causal "
            f"pairs available; sampling all {available} (requested {n_sample})."
        )

    sampled_ids = _stratified_sample(video_event_counts, actual_sample, rng)

    low_count  = sum(1 for v in sampled_ids if video_event_counts[v] < HIGH_EVENT_THRESHOLD)
    high_count = actual_sample - low_count
    print(
        f"  [{dataset_name}] Sampled {actual_sample} videos  "
        f"(low-event: {low_count}, high-event: {high_count})"
    )

    tasks: List[Dict[str, Any]] = []
    clips_dir.mkdir(parents=True, exist_ok=True)

    for video_id in tqdm(sampled_ids, desc=f"  Building clips [{dataset_name}]"):
        video_data = annotations[video_id]
        events     = video_data.get("events", [])
        event_map  = {e["event_id"]: e for e in events}
        edges      = _extract_edges(events)
        n_events   = video_event_counts[video_id]

        video_path = _find_video(video_id, video_dir)
        if video_path is None:
            print(f"  [skip] {video_id}: video not found in {video_dir}")
            continue

        for src_id, dst_id in edges:
            cause_ev  = event_map.get(src_id)
            effect_ev = event_map.get(dst_id)
            if cause_ev is None or effect_ev is None:
                continue

            pair_id   = f"{dataset_name}_{video_id}_{src_id}_{dst_id}"
            clip_name = f"{pair_id}.mp4"
            clip_path = clips_dir / clip_name

            clip_ok = _build_merged_clip(video_path, cause_ev, effect_ev, str(clip_path))

            tasks.append({
                "pair_id":            pair_id,
                "dataset":            dataset_name,
                "video_id":           video_id,
                "cause_event_id":     src_id,
                "effect_event_id":    dst_id,
                "cause_description":  cause_ev["description"],
                "effect_description": effect_ev["description"],
                "cause_timestamp":    cause_ev["timestamp"],
                "effect_timestamp":   effect_ev["timestamp"],
                "n_events_in_video":  n_events,
                "high_event_stratum": n_events >= HIGH_EVENT_THRESHOLD,
                "clip_path":          str(clip_path) if clip_ok else None,
                "clip_ok":            clip_ok,
                # Annotators fill these in (one rating per rater, 1-5 scale)
                "rater_1": None,
                "rater_2": None,
                "rater_3": None,
            })

    return tasks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_tasks(
    actnet_path: str,
    yc2_path: str,
    actnet_video_dir: str,
    yc2_video_dir: str,
    output_dir: str,
) -> None:
    """
    Generate human validation tasks for both ActivityNet and YouCook2.

    Parameters
    ----------
    actnet_path      : Step-3 output JSON for ActivityNet
    yc2_path         : Step-3 output JSON for YouCook2
    actnet_video_dir : root video directory for ActivityNet
    yc2_video_dir    : root video directory for YouCook2
    output_dir       : directory where tasks.json and clips/ are written
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    clips_dir = out / "clips"

    rng = random.Random(RANDOM_SEED)

    print(f"\n[generate_tasks]")
    print(f"  output_dir : {output_dir}")
    print(f"  seed       : {RANDOM_SEED}")

    with open(actnet_path) as f:
        actnet_data: Dict[str, Any] = json.load(f)
    with open(yc2_path) as f:
        yc2_data: Dict[str, Any] = json.load(f)

    print(f"\n  ActivityNet: {len(actnet_data)} entries -> sampling {ACTIVITYNET_SAMPLE} videos")
    actnet_tasks = _generate_tasks_for_dataset(
        actnet_data, actnet_video_dir, "activitynet",
        ACTIVITYNET_SAMPLE, clips_dir, rng,
    )

    print(f"\n  YouCook2: {len(yc2_data)} entries -> sampling {YOUCOOK2_SAMPLE} videos")
    yc2_tasks = _generate_tasks_for_dataset(
        yc2_data, yc2_video_dir, "youcook2",
        YOUCOOK2_SAMPLE, clips_dir, rng,
    )

    all_tasks = actnet_tasks + yc2_tasks
    rng.shuffle(all_tasks)  # Randomise ordering so dataset order doesn't bias annotators

    # Write task manifest
    tasks_path = out / "tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(all_tasks, f, indent=2)

    # Write annotation template CSV (annotators complete this and return it)
    csv_path = out / "annotations_template.csv"
    with open(csv_path, "w") as f:
        f.write("pair_id,rater_1,rater_2,rater_3\n")
        for task in all_tasks:
            f.write(f"{task['pair_id']},,\n")

    # Write sampling statistics
    clip_ok  = sum(1 for t in all_tasks if t["clip_ok"])
    clip_fail = len(all_tasks) - clip_ok
    an_pairs  = sum(1 for t in all_tasks if t["dataset"] == "activitynet")
    yc2_pairs = sum(1 for t in all_tasks if t["dataset"] == "youcook2")
    an_hi     = sum(1 for t in all_tasks if t["dataset"] == "activitynet" and t["high_event_stratum"])
    yc2_hi    = sum(1 for t in all_tasks if t["dataset"] == "youcook2"    and t["high_event_stratum"])

    stats = {
        "total_tasks":           len(all_tasks),
        "clips_created":         clip_ok,
        "clips_failed":          clip_fail,
        "activitynet": {
            "pairs":       an_pairs,
            "high_stratum": an_hi,
            "low_stratum":  an_pairs - an_hi,
        },
        "youcook2": {
            "pairs":       yc2_pairs,
            "high_stratum": yc2_hi,
            "low_stratum":  yc2_pairs - yc2_hi,
        },
        "sampling": {
            "seed":                RANDOM_SEED,
            "high_event_threshold": HIGH_EVENT_THRESHOLD,
            "high_stratum_frac":   HIGH_STRATUM_FRAC,
        },
        "rating_scale": {
            "1": "definitely not causal",
            "2": "probably not causal",
            "3": "uncertain / borderline",
            "4": "probably causal",
            "5": "definitely causal",
        },
        "retention_rule": "mean rating across 3 annotators >= 3.0",
    }

    stats_path = out / "sampling_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Total tasks   : {len(all_tasks)}")
    print(f"  Clips created : {clip_ok} / {len(all_tasks)}")
    print(f"  AN pairs      : {an_pairs}  (high-event: {an_hi})")
    print(f"  YC2 pairs     : {yc2_pairs}  (high-event: {yc2_hi})")
    print(f"\n  tasks.json              -> {tasks_path}")
    print(f"  annotations_template.csv -> {csv_path}")
    print(f"  sampling_stats.json      -> {stats_path}")
    print(f"  clips/                   -> {clips_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    args = parser.parse_args()

    generate_tasks(
        actnet_path      = args.activitynet,
        yc2_path         = args.youcook2,
        actnet_video_dir = args.actnet_video_dir,
        yc2_video_dir    = args.yc2_video_dir,
        output_dir       = args.output_dir,
    )


if __name__ == "__main__":
    main()
