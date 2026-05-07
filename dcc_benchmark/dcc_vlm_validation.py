"""
DCC Benchmark -- Step 3: VLM+LLM Causal Pair Verification
==========================================================
Tier-2 validation of the DCC causal pair annotations produced by Step 2.

For each surviving causal pair (C -> E) from the EMScore-filtered annotations,
this step:
  1. Extracts and merges the cause and effect video segments.
  2. Uniformly samples frames from the merged clip and feeds them to
     InternVL3.5-241B to generate a visual caption grounded in the video.
  3. Scores the causal pair using Llama-3.3-70B across four dimensions
     (plausibility, visual evidence, strength, transition) using the
     structured prompt from Appendix C of the paper.

Implements the validation described in Section 3.2 of the paper:

    "Following [cho2025perceptionlm], we use InternVL3.5-241B to generate
     captions from merged video clips, then evaluate four dimensions via
     Llama-3.3-70B: plausibility, visual evidence, strength, and transition.
     Crucially, Llama judges causal validity from what the VLM sees in the
     merged clips, along with the original human-annotated captions (see
     Appendix C). This targeted grounding is what makes the validation
     reliable despite VLMs' known failure on full-graph causal prediction."

Both the InternVL captioning query and the Llama scoring prompt match
Appendix C (VLM+LLM Validation Prompt) verbatim.

Pipeline
--------
    Step 1 -- dcc_annotation_generator.py
    Step 2 -- dcc_emscore_validation.py
    Step 3 -- dcc_vlm_validation.py  (this file)
    Step 4 -- human_study/
              Tier-3 human validation framework (ICC inter-annotator agreement)

GPU memory note
---------------
    InternVL3.5-241B (~482 GB bfloat16) and Llama-3.3-70B (~140 GB bfloat16)
    are loaded sequentially, not simultaneously.  InternVL is explicitly
    unloaded before Llama is initialised.  On 8x H200 (140 GB each) both
    models fit within the available 1120 GB pool, but loading them together
    leaves little headroom for activations.

Requirements
------------
    pip install transformers torch torchvision pillow opencv-python
    ffmpeg >= 4.0 (must be on PATH)
    InternVL3.5-241B requires >= 4x H200 GPUs (or equivalent)

Usage
-----
    python dcc_vlm_validation.py \\
        --input    dcc_train_emscore.json \\
        --video-dir /data/activitynet/videos \\
        --output   dcc_train_vlm.json

    # YouCook2
    python dcc_vlm_validation.py \\
        --input    dcc_youcook2_emscore.json \\
        --video-dir /data/youcook2/videos \\
        --output   dcc_youcook2_vlm.json
"""

import argparse
import gc
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, pipeline
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

INTERNVL_MODEL_ID = "OpenGVLab/InternVL3_5-241B-A28B"
LLAMA_MODEL_ID    = "meta-llama/Llama-3.3-70B-Instruct"

# Frames uniformly sampled from each merged clip for InternVL input.
N_FRAMES = 8

# Llama overall score threshold for pair retention.
# The paper reports mean validation scores of 0.628 (ActivityNet) and
# 0.632 (YouCook2) after Tier-3 human validation; 0.5 is a conservative floor.
SCORE_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Video utilities  (shared logic with Step 2)
# ---------------------------------------------------------------------------

def _find_video(video_id: str, video_dir: str) -> Optional[str]:
    """
    Locate the raw video file for *video_id* under *video_dir*.

    Tries common naming conventions for ActivityNet and YouCook2 before
    falling back to a recursive glob that handles nested recipe subdirectories.
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
    tmp_dir: str,
    pair_tag: str,
) -> Optional[str]:
    """Extract cause and effect segments and concatenate them in temporal order."""
    ordered = sorted(
        [cause_event, effect_event],
        key=lambda e: e["timestamp"][0],
    )
    segments = []
    for i, event in enumerate(ordered):
        start, end = event["timestamp"]
        seg = os.path.join(tmp_dir, f"{pair_tag}_seg{i}.mp4")
        if not _extract_segment(video_path, start, end, seg):
            return None
        segments.append(seg)

    if len(segments) == 1:
        return segments[0]
    merged = os.path.join(tmp_dir, f"{pair_tag}_merged.mp4")
    return merged if _concat_segments(segments, merged) else None


def _sample_frames(clip_path: str, n_frames: int = N_FRAMES) -> List[Image.Image]:
    """Uniformly sample *n_frames* PIL Images from *clip_path*."""
    cap   = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# InternVL3.5-241B -- image preprocessing
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_INPUT_SIZE    = 448

_FRAME_TRANSFORM = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((_INPUT_SIZE, _INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _preprocess_frames(
    frames: List[Image.Image],
) -> Tuple[torch.Tensor, List[int]]:
    """
    Convert a list of PIL frames into the (pixel_values, num_patches_list) pair
    expected by InternVL's chat / batch_chat API.

    Each frame is treated as a single-tile image (one patch), which avoids
    the complexity of dynamic tiling while still providing temporal coverage
    through frame sampling.
    """
    tensors = [_FRAME_TRANSFORM(f).unsqueeze(0) for f in frames]
    pixel_values    = torch.cat(tensors, dim=0).to(torch.bfloat16)
    num_patches_list = [1] * len(frames)
    return pixel_values, num_patches_list


# ---------------------------------------------------------------------------
# InternVL3.5-241B -- model loading and caption generation
# ---------------------------------------------------------------------------

def _load_internvl(num_gpus: int) -> Tuple[Any, Any]:
    """
    Load InternVL3.5-241B distributed across *num_gpus* GPUs.

    Uses balanced device_map and 140 GB per-GPU memory budget, matching the
    setup used for DCC benchmark annotation (Section 3.2 of the paper).
    """
    print(f"[InternVL] Loading {INTERNVL_MODEL_ID} across {num_gpus} GPU(s) ...")
    model = AutoModel.from_pretrained(
        INTERNVL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="balanced",
        max_memory={i: "140GB" for i in range(num_gpus)},
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        INTERNVL_MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )
    print(f"[InternVL] Model ready.")
    return model, tokenizer


# Captioning prompt fed to InternVL for each merged clip.
# The model is asked to describe visual content without any causal framing so
# that Llama's subsequent scoring is grounded in an unbiased visual description.
_INTERNVL_CAPTION_PROMPT = (
    "Describe what is happening in this sequence of video frames. "
    "Be specific about the actions, objects, and activities you observe. "
    "Keep the description factual and concise."
)


def _caption_clip(
    model: Any,
    tokenizer: Any,
    frames: List[Image.Image],
) -> Optional[str]:
    """
    Generate a single caption for the *frames* of a merged clip.

    Returns the caption string, or None if generation fails.
    """
    if not frames:
        return None
    try:
        pixel_values, num_patches_list = _preprocess_frames(frames)
        pixel_values = pixel_values.cuda()
        # One <image> token per frame, consistent with InternVL multi-image chat
        image_tokens = "\n".join("<image>" for _ in frames)
        question     = f"{image_tokens}\n{_INTERNVL_CAPTION_PROMPT}"
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            dict(max_new_tokens=256, do_sample=False),
            num_patches_list=num_patches_list,
        )
        return response
    except Exception as exc:
        print(f"  [InternVL] Caption generation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Llama-3.3-70B -- scoring with the Appendix C prompt
# ---------------------------------------------------------------------------

_LLAMA_SYSTEM_PROMPT = (
    "You are evaluating whether a proposed causal relationship is supported "
    "by visual evidence in a video."
)

# Output schema shown verbatim in Appendix C of the paper
_SCORE_JSON_SCHEMA = """{
    "plausibility": float,
    "visual_evidence": float,
    "strength": float,
    "transition": float,
    "overall": float,
    "reasoning": "brief explanation (max 50 words)"
}"""


def _build_scoring_prompt(
    cause_descriptions: str,
    effect_descriptions: str,
    generated_caption: str,
) -> str:
    """
    Construct the user-turn scoring prompt.

    Matches the VLM+LLM Validation Prompt in Appendix C of the paper verbatim,
    substituting {cause_descriptions}, {effect_descriptions}, and
    {generated_caption} with the actual values for each pair.
    """
    return (
        f"Given:\n"
        f"- Cause events: {cause_descriptions}\n"
        f"- Effect events: {effect_descriptions}\n"
        f"- Generated video caption: {generated_caption}\n\n"
        "Evaluate the following four aspects (score 0.0-1.0 for each):\n"
        "1. PLAUSIBILITY (0.0-1.0):\n"
        "Does the generated caption make the proposed causal link plausible?\n"
        "- 0.0: Caption contradicts the causal claim\n"
        "- 0.5: Caption is neutral, neither supports nor contradicts\n"
        "- 1.0: Caption strongly supports the causal claim\n"
        "2. VISUAL_EVIDENCE (0.0-1.0):\n"
        "How much explicit or implicit visual evidence for causality appears?\n"
        "- 0.0: No evidence of cause or effect visible\n"
        "- 0.5: Both cause and effect visible, but connection unclear\n"
        "- 1.0: Clear visual evidence of causal mechanism\n"
        "3. STRENGTH (0.0-1.0):\n"
        "How direct and strong is the causal connection?\n"
        "Consider necessity (effect requires cause) and sufficiency "
        "(cause typically produces effect).\n"
        "- 0.0: Events appear independent\n"
        "- 0.5: Weak or indirect connection\n"
        "- 1.0: Strong, direct causal relationship\n"
        "4. TRANSITION (0.0-1.0):\n"
        "How smooth and coherent is the temporal progression from cause to effect?\n"
        "- 0.0: Abrupt or illogical transition\n"
        "- 0.5: Acceptable temporal flow\n"
        "- 1.0: Smooth, natural progression\n"
        "Guidelines:\n"
        "- Base judgment ONLY on provided information\n"
        "- Do not introduce external knowledge\n"
        "- Consider both what is shown and what is implied\n"
        "- Be objective and consistent\n\n"
        f"Output JSON format:\n{_SCORE_JSON_SCHEMA}\n\n"
        "Provide only the JSON object output."
    )


def _load_llama(gpu_id: int = 0) -> Any:
    """Load Llama-3.3-70B on a single GPU for sequential scoring."""
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"[Llama] Loading {LLAMA_MODEL_ID} on {device} ...")
    pipe = pipeline(
        "text-generation",
        model=LLAMA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device=device,
        model_kwargs={"low_cpu_mem_usage": True},
    )
    pipe.tokenizer.padding_side = "left"
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    print(f"[Llama] Model ready.")
    return pipe


def _score_pairs_llama(
    pipe: Any,
    scoring_inputs: List[Dict[str, str]],
    batch_size: int = 4,
) -> List[Optional[Dict[str, Any]]]:
    """
    Score a list of causal pairs with Llama-3.3-70B.

    Parameters
    ----------
    scoring_inputs : list of dicts with keys
                     cause_descriptions, effect_descriptions, generated_caption
    batch_size     : number of pairs per forward pass

    Returns
    -------
    list of score dicts (or None on parse failure), one per input
    """
    all_scores: List[Optional[Dict[str, Any]]] = []

    for start in range(0, len(scoring_inputs), batch_size):
        batch = scoring_inputs[start: start + batch_size]
        messages_batch = [
            [
                {"role": "system", "content": _LLAMA_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_scoring_prompt(
                        inp["cause_descriptions"],
                        inp["effect_descriptions"],
                        inp["generated_caption"],
                    ),
                },
            ]
            for inp in batch
        ]

        responses = pipe(
            messages_batch,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
            batch_size=batch_size,
        )

        for response in responses:
            text  = response[0]["generated_text"]
            start_idx = text.find("{")
            if start_idx == -1:
                all_scores.append(None)
                continue
            try:
                obj, _ = json.JSONDecoder().raw_decode(text, start_idx)
                all_scores.append(obj)
            except json.JSONDecodeError:
                all_scores.append(None)

    return all_scores


# ---------------------------------------------------------------------------
# Annotation edge utilities
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
    Prune causal chains to only the validated edges and recompute
    is_cause / is_effect / event_type flags.
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
# Phase 1: InternVL caption generation
# ---------------------------------------------------------------------------

def _phase1_generate_captions(
    data: Dict[str, Any],
    video_dir: str,
    num_gpus: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate a visual caption for every causal pair using InternVL3.5-241B.

    For each pair (cause_id -> effect_id), the corresponding video segments
    are extracted and merged; N_FRAMES frames are sampled and fed to InternVL.

    Returns
    -------
    pair_captions : nested dict  { video_id -> { "pair_{src}_{dst}" -> {...} } }
        Each inner dict has keys:
            cause_description  : event description from the DCC annotation
            effect_description : event description from the DCC annotation
            caption            : InternVL-generated caption (None on failure)
    """
    model, tokenizer = _load_internvl(num_gpus)
    pair_captions: Dict[str, Dict[str, Any]] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for video_id, video_data in tqdm(data.items(), desc="Phase 1 [InternVL captions]"):
            if video_data is None:
                continue

            events    = video_data.get("events", [])
            event_map = {e["event_id"]: e for e in events}
            edges     = _extract_edges(events)
            if not edges:
                continue

            video_path = _find_video(video_id, video_dir)
            if video_path is None:
                print(f"  [skip] {video_id}: video not found in {video_dir}")
                continue

            pair_captions[video_id] = {}

            for src_id, dst_id in edges:
                cause_ev  = event_map.get(src_id)
                effect_ev = event_map.get(dst_id)
                if cause_ev is None or effect_ev is None:
                    continue

                pair_tag  = f"{video_id}_{src_id}_{dst_id}"
                clip_path = _build_merged_clip(
                    video_path, cause_ev, effect_ev, tmp_dir, pair_tag
                )
                frames  = _sample_frames(clip_path) if clip_path else []
                caption = _caption_clip(model, tokenizer, frames)

                pair_captions[video_id][f"pair_{src_id}_{dst_id}"] = {
                    "cause_description":  cause_ev["description"],
                    "effect_description": effect_ev["description"],
                    "caption":            caption,
                }

    # Unload InternVL before loading Llama
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return pair_captions


# ---------------------------------------------------------------------------
# Phase 2: Llama scoring
# ---------------------------------------------------------------------------

def _phase2_score_captions(
    pair_captions: Dict[str, Dict[str, Any]],
    llama_gpu: int,
    batch_size: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Score all InternVL captions with Llama-3.3-70B using the Appendix C prompt.

    Pairs whose caption is None (InternVL failure) are skipped; their score
    entry is left absent so the caller can handle them conservatively.
    """
    pipe = _load_llama(llama_gpu)

    # Collect all scorable pairs in order so we can map responses back
    keys:   List[Tuple[str, str]]   = []
    inputs: List[Dict[str, str]]    = []

    for video_id, pairs in pair_captions.items():
        for pair_key, info in pairs.items():
            if info.get("caption") is None:
                continue
            keys.append((video_id, pair_key))
            inputs.append({
                "cause_descriptions":  info["cause_description"],
                "effect_descriptions": info["effect_description"],
                "generated_caption":   info["caption"],
            })

    scores = _score_pairs_llama(pipe, inputs, batch_size=batch_size)

    for (video_id, pair_key), score in zip(keys, scores):
        pair_captions[video_id][pair_key]["scores"] = score

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return pair_captions


# ---------------------------------------------------------------------------
# Merge scores back into annotation structure
# ---------------------------------------------------------------------------

def _apply_scores_and_filter(
    data: Dict[str, Any],
    pair_captions: Dict[str, Dict[str, Any]],
    threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Remove causal edges whose Llama overall score is below *threshold*.

    Edges for which InternVL caption generation failed are kept conservatively
    (they already passed Tier-1 EMScore validation).

    Returns
    -------
    (filtered_annotations, per_video_report)
    """
    filtered: Dict[str, Any] = {}
    report:   Dict[str, Any] = {}

    for video_id, video_data in data.items():
        if video_data is None:
            filtered[video_id] = None
            continue

        events    = video_data.get("events", [])
        edges     = _extract_edges(events)
        pairs     = pair_captions.get(video_id, {})

        valid_edges: Set[Tuple[int, int]] = set()
        pair_report: List[Dict[str, Any]] = []

        for src_id, dst_id in edges:
            info   = pairs.get(f"pair_{src_id}_{dst_id}", {})
            scores = info.get("scores")

            if scores is None:
                # Caption generation failed; keep pair (conservative)
                valid_edges.add((src_id, dst_id))
                pair_report.append({
                    "pair":   [src_id, dst_id],
                    "valid":  True,
                    "reason": "caption_unavailable_kept",
                })
                continue

            overall = float(scores.get("overall", 0.0))
            valid   = overall >= threshold
            if valid:
                valid_edges.add((src_id, dst_id))

            pair_report.append({
                "pair":    [src_id, dst_id],
                "scores":  scores,
                "overall": round(overall, 4),
                "valid":   valid,
            })

        updated_events = _rebuild_chains(events, valid_edges)
        filtered[video_id] = {**video_data, "events": updated_events}
        report[video_id]   = {
            "total_pairs": len(edges),
            "valid_pairs": len(valid_edges),
            "pair_scores": pair_report,
        }

    return filtered, report


# ---------------------------------------------------------------------------
# Dataset-level entry point
# ---------------------------------------------------------------------------

def validate_dataset(
    input_path: str,
    video_dir: str,
    output_path: str,
    report_path: str,
    num_gpus: int,
    score_threshold: float,
    llama_gpu: int,
    llama_batch_size: int,
) -> None:
    """
    Run Tier-2 VLM+LLM validation over the DCC annotation file.

    Parameters
    ----------
    input_path       : EMScore-filtered annotation JSON from dcc_emscore_validation.py
    video_dir        : directory containing raw video files
    output_path      : output path for the Tier-2-filtered annotation JSON
    report_path      : output path for per-video Llama scoring statistics
    num_gpus         : number of GPUs for InternVL3.5-241B (>= 4 recommended)
    score_threshold  : Llama overall score threshold for pair retention
    llama_gpu        : GPU index for Llama-3.3-70B (loaded after InternVL is freed)
    llama_batch_size : forward-pass batch size for Llama scoring
    """
    print(f"\n[validate_dataset]")
    print(f"  input           : {input_path}")
    print(f"  video_dir       : {video_dir}")
    print(f"  score_threshold : {score_threshold}")
    print(f"  InternVL GPUs   : {num_gpus}")
    print(f"  Llama GPU       : {llama_gpu}")

    with open(input_path) as f:
        data: Dict[str, Any] = json.load(f)

    valid_entries = {vid: d for vid, d in data.items() if d is not None}
    print(f"  Loaded {len(valid_entries)} annotated videos.")

    # Phase 1: InternVL3.5-241B generates captions
    pair_captions = _phase1_generate_captions(valid_entries, video_dir, num_gpus)

    # Phase 2: Llama-3.3-70B scores captions (InternVL already unloaded)
    pair_captions = _phase2_score_captions(pair_captions, llama_gpu, llama_batch_size)

    # Filter edges by score threshold and rebuild annotation chains
    filtered, report = _apply_scores_and_filter(valid_entries, pair_captions, score_threshold)

    total_pairs = sum(r["total_pairs"] for r in report.values())
    total_valid = sum(r["valid_pairs"] for r in report.values())
    retention   = total_valid / total_pairs if total_pairs else 0.0

    all_scores = [
        entry["overall"]
        for r in report.values()
        for entry in r["pair_scores"]
        if isinstance(entry.get("overall"), float)
    ]
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    aggregate = {
        "total_videos":    len(valid_entries),
        "total_pairs":     total_pairs,
        "valid_pairs":     total_valid,
        "invalid_pairs":   total_pairs - total_valid,
        "retention_rate":  round(retention, 4),
        "mean_overall_score": round(mean_score, 4),
        "score_threshold": score_threshold,
    }
    print(
        f"\n  Retained {total_valid}/{total_pairs} pairs "
        f"({100 * retention:.1f}%)  |  mean score: {mean_score:.3f}"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    with open(report_path, "w") as f:
        json.dump({"aggregate": aggregate, "per_video": report}, f, indent=2)

    print(f"  Filtered annotations -> {output_path}")
    print(f"  Scoring report       -> {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    n_available = torch.cuda.device_count() if torch.cuda.is_available() else 1

    parser = argparse.ArgumentParser(
        description="DCC benchmark -- VLM+LLM causal pair verification (Step 3 of 3)"
    )
    parser.add_argument(
        "--input", required=True, metavar="FILE",
        help="EMScore-filtered DCC annotation JSON from dcc_emscore_validation.py",
    )
    parser.add_argument(
        "--video-dir", required=True, metavar="DIR",
        help=(
            "Root directory containing raw video files. "
            "Both flat (ActivityNet) and nested recipe subdirectories "
            "(YouCook2) are resolved automatically."
        ),
    )
    parser.add_argument(
        "--output", required=True, metavar="FILE",
        help="Output path for the Tier-2-filtered annotation JSON",
    )
    parser.add_argument(
        "--report", default=None, metavar="FILE",
        help="Output path for per-video scoring statistics (default: <output stem>_report.json)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=n_available,
        help=f"Number of GPUs for InternVL3.5-241B (default: {n_available})",
    )
    parser.add_argument(
        "--llama-gpu", type=int, default=0,
        help="GPU index for Llama-3.3-70B loaded after InternVL is freed (default: 0)",
    )
    parser.add_argument(
        "--llama-batch-size", type=int, default=4,
        help="Forward-pass batch size for Llama scoring (default: 4)",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=SCORE_THRESHOLD,
        help=f"Llama overall score threshold for pair retention (default: {SCORE_THRESHOLD})",
    )
    args = parser.parse_args()

    report_path = (
        args.report
        if args.report is not None
        else str(Path(args.output).with_suffix("")) + "_report.json"
    )

    validate_dataset(
        input_path       = args.input,
        video_dir        = args.video_dir,
        output_path      = args.output,
        report_path      = report_path,
        num_gpus         = args.num_gpus,
        score_threshold  = args.score_threshold,
        llama_gpu        = args.llama_gpu,
        llama_batch_size = args.llama_batch_size,
    )


if __name__ == "__main__":
    main()
