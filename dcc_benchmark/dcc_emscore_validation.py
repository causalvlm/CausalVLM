"""
DCC Benchmark -- Step 2: EMScore Multimodal Alignment Validation
================================================================
Tier-1 validation of the DCC causal pair annotations produced by Step 1.

For each proposed causal pair (cause event C -> effect event E), the
corresponding video segments are extracted and merged into a single clip.
EMScore(X,V) is then computed between the merged clip and the concatenated
event descriptions.  Pairs whose score falls below the threshold theta=0.2
are removed from the annotation.

Implements the validation described in Section 3.2 of the paper:

    "For each causal pair (C, E), we merge video segments and compute
     EMScore (theta=0.2, following NarrativeBridge). This validates temporal
     coherence: 94.3% of pairs exceed threshold on ActivityNet, 91.8% on
     YouCook2."

Pipeline
--------
    Step 1 -- dcc_annotation_generator.py
              LLM causal annotation via Llama-3.3-70B
    Step 2 -- dcc_emscore_validation.py  (this file)
              Tier-1 multimodal alignment via EMScore
    Step 3 -- dcc_vlm_validation.py
              Tier-2 VLM+LLM causal-pair verification
              (InternVL3.5-241B caption generation + Llama-3.3-70B scoring)
    Step 4 -- human_study/
              Tier-3 human validation framework (ICC inter-annotator agreement)

Requirements
------------
    pip install emscore
    ffmpeg >= 4.0 (must be on PATH)

Input
-----
    annotation JSON produced by dcc_annotation_generator.py
    video directory for ActivityNet or YouCook2

    ActivityNet videos are expected as flat files:
        <video-dir>/<video_id>.mp4   or   <video-dir>/v_<video_id>.mp4
    YouCook2 videos may be in recipe subdirectories:
        <video-dir>/<recipe_id>/<video_id>.mp4
    Both layouts are found automatically via recursive glob.

Output
------
    <output>.json        filtered annotations (same schema as input,
                         causal edges below threshold removed,
                         chain fields and event_type flags updated)
    <output>_report.json per-video and aggregate EMScore statistics

Usage
-----
    # ActivityNet
    python dcc_emscore_validation.py \\
        --input  dcc_train.json \\
        --video-dir /data/activitynet/videos \\
        --output dcc_train_emscore.json

    # YouCook2
    python dcc_emscore_validation.py \\
        --input  dcc_youcook2.json \\
        --video-dir /data/youcook2/videos \\
        --output dcc_youcook2_emscore.json
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from emscore.scorer import EMScorer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Validation threshold theta from Section 3.2 of the paper.
# Follows NarrativeBridge (Nadeem et al., 2025).
EMSCORE_THRESHOLD: float = 0.2


# ---------------------------------------------------------------------------
# Video file utilities
# ---------------------------------------------------------------------------

def _find_video(video_id: str, video_dir: str) -> Optional[str]:
    """
    Locate the raw video file for *video_id* under *video_dir*.

    Tries common naming patterns for ActivityNet and YouCook2 before
    falling back to a recursive glob so that nested recipe directories
    (YouCook2) are handled transparently.

    Returns the path string if found, else None.
    """
    base = Path(video_dir)
    candidates = [
        base / f"{video_id}.mp4",
        base / f"v_{video_id}.mp4",
        base / f"{video_id}.mkv",
        base / f"{video_id}.webm",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    # Recursive search covers YouCook2 recipe subdirectories
    matches = sorted(base.rglob(f"{video_id}.*"))
    return str(matches[0]) if matches else None


def _extract_segment(
    video_path: str,
    start: float,
    end: float,
    out_path: str,
) -> bool:
    """
    Extract the video segment [start, end] seconds into *out_path*.

    Placing -ss before -i uses keyframe-accurate input seeking which is
    faster and frame-accurate enough for EMScore computation.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        out_path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _concat_segments(segment_paths: List[str], out_path: str) -> bool:
    """
    Concatenate video segments (in the given order) into *out_path* using
    the ffmpeg concat demuxer.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        for seg in segment_paths:
            f.write(f"file '{os.path.abspath(seg)}'\n")
        concat_list = f.name

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        out_path,
    ]
    ok = subprocess.run(cmd, capture_output=True).returncode == 0
    os.unlink(concat_list)
    return ok


def _build_merged_clip(
    video_path: str,
    cause_event: Dict[str, Any],
    effect_event: Dict[str, Any],
    tmp_dir: str,
    pair_tag: str,
) -> Optional[str]:
    """
    Extract cause and effect segments from *video_path* and concatenate them
    in temporal order.

    Returns the path to the merged clip, or None if any ffmpeg step fails.
    """
    # Sort by start time so cause always precedes effect in the merged clip
    ordered = sorted(
        [cause_event, effect_event],
        key=lambda e: e["timestamp"][0],
    )

    segment_paths = []
    for i, event in enumerate(ordered):
        start, end = event["timestamp"]
        seg_path = os.path.join(tmp_dir, f"{pair_tag}_seg{i}.mp4")
        if not _extract_segment(video_path, start, end, seg_path):
            return None
        segment_paths.append(seg_path)

    if len(segment_paths) == 1:
        return segment_paths[0]

    merged_path = os.path.join(tmp_dir, f"{pair_tag}_merged.mp4")
    return merged_path if _concat_segments(segment_paths, merged_path) else None


# ---------------------------------------------------------------------------
# EMScore computation
# ---------------------------------------------------------------------------

def _build_caption(cause_event: Dict[str, Any], effect_event: Dict[str, Any]) -> str:
    """
    Build the candidate caption for an (C, E) pair by concatenating the
    event descriptions in temporal order, matching the text fed to the LLM
    in Step 1.
    """
    ordered = sorted(
        [cause_event, effect_event],
        key=lambda e: e["timestamp"][0],
    )
    return " ".join(e["description"] for e in ordered)


def _emscore_xv(clip_path: str, caption: str, scorer: EMScorer) -> float:
    """
    Compute EMScore(X,V) between *caption* and *clip_path*.

    Returns the full_F score (harmonic mean of precision and recall).
    The same caption is passed as the text reference so that the API is
    satisfied; only the video-grounded EMScore(X,V) metric is used.
    """
    results = scorer.score(
        cands=[caption],
        refs=[caption],
        vids=[clip_path],
        idf=False,
    )
    return float(results["EMScore(X,V)"]["full_F"])


# ---------------------------------------------------------------------------
# Annotation filtering
# ---------------------------------------------------------------------------

def _extract_edges(events: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Return all directed causal edges (cause_id, effect_id) in the annotation."""
    edges = []
    for event in events:
        src = event["event_id"]
        for dst in event["chain"].get("effect_event", []):
            edges.append((src, int(dst)))
    return edges


def _rebuild_chains(
    events: List[Dict[str, Any]],
    valid_edges: Set[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """
    Reconstruct the chain fields for every event using only the validated
    edges, and recompute event_type accordingly.

    event_type assignment:
        "cause"       -- only causes other events
        "effect"      -- only caused by other events
        "causal"      -- both causes and is caused (bidirectional mediator)
        "independent" -- no causal links
    """
    ids = {e["event_id"] for e in events}

    new_effect: Dict[int, List[int]] = {e["event_id"]: [] for e in events}
    new_cause:  Dict[int, List[int]] = {e["event_id"]: [] for e in events}

    for src, dst in valid_edges:
        if src in ids and dst in ids:
            new_effect[src].append(dst)
            new_cause[dst].append(src)

    updated = []
    for event in events:
        eid       = event["event_id"]
        effect_ev = sorted(new_effect[eid])
        cause_ev  = sorted(new_cause[eid])
        is_cause  = len(effect_ev) > 0
        is_effect = len(cause_ev) > 0

        if is_cause and is_effect:
            event_type = "causal"
        elif is_cause:
            event_type = "cause"
        elif is_effect:
            event_type = "effect"
        else:
            event_type = "independent"

        updated.append({
            **event,
            "chain": {
                "is_cause":    is_cause,
                "is_effect":   is_effect,
                "cause_event": cause_ev,
                "effect_event": effect_ev,
            },
            "event_type": event_type,
        })
    return updated


# ---------------------------------------------------------------------------
# Per-video validation
# ---------------------------------------------------------------------------

def _validate_video(
    video_id: str,
    video_data: Dict[str, Any],
    video_dir: str,
    scorer: EMScorer,
    tmp_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate all causal pairs for one video against its raw video file.

    Parameters
    ----------
    video_id   : identifier used to locate the video file
    video_data : annotation dict for this video (events + duration)
    video_dir  : root directory for video files
    scorer     : shared EMScorer instance
    tmp_dir    : temporary directory for intermediate clip files

    Returns
    -------
    (updated_video_data, stats_dict)
        updated_video_data : annotation with invalid edges removed
        stats_dict         : per-pair scores and aggregate counts
    """
    events    = video_data.get("events", [])
    event_map = {e["event_id"]: e for e in events}
    edges     = _extract_edges(events)

    stats: Dict[str, Any] = {
        "total_pairs":   len(edges),
        "valid_pairs":   0,
        "skipped_pairs": 0,
        "pair_scores":   [],
    }

    if not edges:
        return video_data, stats

    video_path = _find_video(video_id, video_dir)
    if video_path is None:
        print(f"  [skip] {video_id}: video not found in {video_dir}")
        stats["skipped_pairs"] = len(edges)
        return video_data, stats

    valid_edges: Set[Tuple[int, int]] = set()

    for pair_idx, (src_id, dst_id) in enumerate(edges):
        cause_ev  = event_map.get(src_id)
        effect_ev = event_map.get(dst_id)

        if cause_ev is None or effect_ev is None:
            stats["skipped_pairs"] += 1
            continue

        pair_tag  = f"{video_id}_{pair_idx}"
        clip_path = _build_merged_clip(
            video_path, cause_ev, effect_ev, tmp_dir, pair_tag
        )
        if clip_path is None:
            stats["skipped_pairs"] += 1
            continue

        caption = _build_caption(cause_ev, effect_ev)
        score   = _emscore_xv(clip_path, caption, scorer)

        stats["pair_scores"].append({
            "cause_event_id":  src_id,
            "effect_event_id": dst_id,
            "emscore":         round(score, 4),
            "valid":           score >= EMSCORE_THRESHOLD,
        })

        if score >= EMSCORE_THRESHOLD:
            valid_edges.add((src_id, dst_id))
            stats["valid_pairs"] += 1

    updated_events = _rebuild_chains(events, valid_edges)
    updated_data   = {**video_data, "events": updated_events}
    return updated_data, stats


# ---------------------------------------------------------------------------
# Dataset-level entry point
# ---------------------------------------------------------------------------

def validate_dataset(
    input_path: str,
    video_dir: str,
    output_path: str,
    report_path: str,
) -> None:
    """
    Run Tier-1 EMScore validation over the full DCC annotation file.

    Parameters
    ----------
    input_path  : DCC annotation JSON from dcc_annotation_generator.py
    video_dir   : directory containing the raw video files
    output_path : path for the filtered annotation JSON
    report_path : path for the per-video and aggregate statistics JSON
    """
    print(f"\n[validate_dataset]")
    print(f"  input     : {input_path}")
    print(f"  video_dir : {video_dir}")
    print(f"  threshold : theta={EMSCORE_THRESHOLD}")

    with open(input_path) as f:
        data: Dict[str, Any] = json.load(f)

    valid_entries = {vid: d for vid, d in data.items() if d is not None}
    print(f"  Loaded {len(valid_entries)} annotated videos ({len(data)} total entries).")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"  EMScorer  : {device}")
    scorer = EMScorer(device=device)

    filtered: Dict[str, Any] = {}
    report:   Dict[str, Any] = {}
    total_pairs = 0
    total_valid = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for video_id, video_data in tqdm(valid_entries.items(), desc="EMScore validation"):
            updated, stats = _validate_video(
                video_id, video_data, video_dir, scorer, tmp_dir
            )
            filtered[video_id] = updated
            report[video_id]   = stats
            total_pairs += stats["total_pairs"]
            total_valid += stats["valid_pairs"]

    retention = total_valid / total_pairs if total_pairs else 0.0
    aggregate = {
        "total_videos":  len(valid_entries),
        "total_pairs":   total_pairs,
        "valid_pairs":   total_valid,
        "invalid_pairs": total_pairs - total_valid,
        "retention_rate": round(retention, 4),
        "threshold":     EMSCORE_THRESHOLD,
    }
    print(
        f"\n  Retained {total_valid}/{total_pairs} pairs "
        f"({100 * retention:.1f}%)"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    with open(report_path, "w") as f:
        json.dump({"aggregate": aggregate, "per_video": report}, f, indent=2)

    print(f"  Filtered annotations -> {output_path}")
    print(f"  Validation report    -> {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DCC benchmark -- EMScore multimodal alignment validation (Step 2 of 3)"
        )
    )
    parser.add_argument(
        "--input", required=True, metavar="FILE",
        help="DCC annotation JSON from dcc_annotation_generator.py",
    )
    parser.add_argument(
        "--video-dir", required=True, metavar="DIR",
        help=(
            "Root directory containing raw video files. "
            "Both flat layouts (ActivityNet) and nested recipe subdirectories "
            "(YouCook2) are resolved automatically."
        ),
    )
    parser.add_argument(
        "--output", required=True, metavar="FILE",
        help="Output path for the filtered annotation JSON",
    )
    parser.add_argument(
        "--report", default=None, metavar="FILE",
        help=(
            "Output path for per-video validation statistics "
            "(default: <output stem>_report.json)"
        ),
    )
    args = parser.parse_args()

    report_path = (
        args.report
        if args.report is not None
        else str(Path(args.output).with_suffix("")) + "_report.json"
    )

    validate_dataset(args.input, args.video_dir, args.output, report_path)


if __name__ == "__main__":
    main()
