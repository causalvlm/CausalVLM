"""Validate DCC pairs with InternVL captions and Llama scores. Requires an explicit retention cutoff."""

import argparse
import gc
import json
import os
import subprocess
import tempfile
from pathlib import Path
from media import find_video as _find_video
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from schema import normalize_scores, validate_annotations, require_new_paths, write_json
from transformers import AutoModel, AutoTokenizer, pipeline, set_seed
from tqdm import tqdm

# Model identifiers

INTERNVL_MODEL_ID = "OpenGVLab/InternVL3_5-241B-A28B"
LLAMA_MODEL_ID    = "meta-llama/Llama-3.3-70B-Instruct"

# Frames uniformly sampled from each merged clip for InternVL input.
N_FRAMES = 8

# Video utilities  (shared logic with Step 2)



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


# InternVL3.5-241B -- image preprocessing

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

    Each sampled frame is represented by one image tile.
    """
    tensors = [_FRAME_TRANSFORM(f).unsqueeze(0) for f in frames]
    pixel_values    = torch.cat(tensors, dim=0).to(torch.bfloat16)
    num_patches_list = [1] * len(frames)
    return pixel_values, num_patches_list


# InternVL3.5-241B -- model loading and caption generation

def _gpu_memory(num_gpus):
    if not 1 <= num_gpus <= torch.cuda.device_count():
        raise ValueError('num-gpus must be positive and no larger than the visible GPU count')
    # Reserve 10% of free GPU memory for activations and the KV cache.
    return {i: int(torch.cuda.mem_get_info(i)[0] * .90) for i in range(num_gpus)}


def _load_internvl(num_gpus, model_id=INTERNVL_MODEL_ID, flash_attention=False):
    """Shard the model using the actual free memory of the allocated GPUs."""
    print(f'[InternVL] Loading {model_id} across {num_gpus} GPU(s) ...', flush=True)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=flash_attention, trust_remote_code=True,
        device_map='balanced', max_memory=_gpu_memory(num_gpus)).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    print('[InternVL] Model ready.', flush=True)
    return model, tokenizer


# Request a visual description without supplying the proposed causal labels.
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
        pixel_values = pixel_values.to(next(model.vision_model.parameters()).device)
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
        return response if isinstance(response, str) and response.strip() else None
    except Exception as exc:
        print(f"  [InternVL] Caption generation failed: {exc}")
        return None


# Llama-3.3-70B -- scoring with the Appendix C prompt

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
    "overall": float (average of four scores),
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


def _load_llama(gpu_id=None, model_id=LLAMA_MODEL_ID, num_gpus=None):
    """Shard the scoring model after InternVL is unloaded."""
    memory = _gpu_memory(num_gpus or torch.cuda.device_count())
    if gpu_id is not None:
        if gpu_id not in memory:
            raise ValueError('llama-gpu must identify a visible, allocated GPU')
        memory = {gpu_id: memory[gpu_id]}
    print(f'[Llama] Loading {model_id} across GPU(s) {list(memory)} ...', flush=True)
    pipe = pipeline('text-generation', model=model_id, torch_dtype=torch.bfloat16,
                    device_map='auto', model_kwargs={'low_cpu_mem_usage': True, 'max_memory': memory})
    pipe.tokenizer.padding_side = 'left'
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    return pipe


def _score_pairs_llama(
    pipe: Any,
    scoring_inputs: List[Dict[str, str]],
    batch_size: int = 1,
    raw_file=None,
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
    if batch_size < 1:
        raise ValueError("Scoring batch size must be positive")
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

        if len(responses) != len(batch):
            raise RuntimeError('Scoring response count differs from input batch')
        for inp, response in zip(batch, responses):
            text = response[0]['generated_text']
            if raw_file is not None:
                raw_file.write(json.dumps({'input': inp, 'raw_response': text}, allow_nan=False) + '\n')
                raw_file.flush()
            start_idx = text.find("{")
            if start_idx == -1:
                all_scores.append(None)
                continue
            try:
                obj, _ = json.JSONDecoder().raw_decode(text, start_idx)
                all_scores.append(normalize_scores(obj))
            except (json.JSONDecodeError, ValueError, TypeError):
                all_scores.append(None)

    return all_scores


# Annotation edge utilities

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


# Phase 1: InternVL caption generation

def _phase1_generate_captions(
    data: Dict[str, Any],
    video_dir: str,
    num_gpus: int,
    model_id=INTERNVL_MODEL_ID,
    flash_attention=False,
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
    if not any(_extract_edges(record["events"]) for record in data.values()):
        return {}
    model, tokenizer = _load_internvl(num_gpus, model_id, flash_attention)
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
                with tempfile.TemporaryDirectory(dir=tmp_dir) as pair_tmp:
                    clip_path = _build_merged_clip(
                        video_path, cause_ev, effect_ev, pair_tmp, pair_tag
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


# Phase 2: Llama scoring

def _phase2_score_captions(
    pair_captions: Dict[str, Dict[str, Any]],
    llama_gpu: int,
    batch_size: int,
    model_id=LLAMA_MODEL_ID,
    num_gpus=None,
    raw_path=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Score all InternVL captions with Llama-3.3-70B using the Appendix C prompt.

    Pairs whose caption is None (InternVL failure) are skipped; their score
    entry remains absent and blocks publication of final labels.
    """
    # Retain input order when mapping model responses to pairs.
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

    if not inputs:
        return pair_captions
    pipe = _load_llama(llama_gpu, model_id, num_gpus)
    if raw_path is None:
        scores = _score_pairs_llama(pipe, inputs, batch_size=batch_size)
    else:
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, 'x') as raw:
            scores = _score_pairs_llama(pipe, inputs, batch_size=batch_size, raw_file=raw)

    for (video_id, pair_key), score in zip(keys, scores):
        pair_captions[video_id][pair_key]["scores"] = score

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return pair_captions


# Merge scores back into annotation structure

def _apply_scores_and_filter(
    data: Dict[str, Any],
    pair_captions: Dict[str, Dict[str, Any]],
    threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Remove causal edges whose mean component score is below *threshold*.

    Unscored edges are marked incomplete. The dataset writer refuses to release
    filtered annotations until every proposed pair has been validated.

    Returns
    -------
    (filtered_annotations, per_video_report)
    """
    if not 0 <= threshold <= 1:
        raise ValueError("score-threshold must be in [0, 1]")
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
            if scores is not None:
                try:
                    scores = normalize_scores(scores)
                except (ValueError, TypeError):
                    scores = None

            if scores is None:
                # Preserve the input edge internally, but block final output below.
                valid_edges.add((src_id, dst_id))
                pair_report.append({
                    "pair":   [src_id, dst_id],
                    "valid":  None,
                    "reason": "validation_unavailable",
                })
                continue

            overall = scores["overall"]
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
            "valid_pairs": sum(p["valid"] is True for p in pair_report),
            "unvalidated_pairs": sum(p["valid"] is None for p in pair_report),
            "pair_scores": pair_report,
        }

    return filtered, report


# Dataset-level entry point

def validate_dataset(
    input_path: str,
    video_dir: str,
    output_path: str,
    report_path: str,
    num_gpus: int,
    score_threshold: float,
    llama_gpu: int,
    llama_batch_size: int,
    internvl_model=INTERNVL_MODEL_ID,
    llama_model=LLAMA_MODEL_ID,
    flash_attention=False,
    seed=42,
) -> None:
    """
    Run Tier-2 VLM+LLM validation over the DCC annotation file.

    Parameters
    ----------
    input_path       : EMScore-filtered annotation JSON from dcc_emscore_validation.py
    video_dir        : directory containing raw video files
    output_path      : output path for the Tier-2-filtered annotation JSON
    report_path      : output path for per-video Llama scoring statistics
    num_gpus         : allocated GPUs for both models (four H200s tested)
    score_threshold  : mean component score threshold for pair retention
    llama_gpu        : optional single GPU; None shards after InternVL is freed
    llama_batch_size : forward-pass batch size for Llama scoring
    """
    print(f"\n[validate_dataset]")
    print(f"  input           : {input_path}")
    print(f"  video_dir       : {video_dir}")
    print(f"  score_threshold : {score_threshold}")
    print(f"  InternVL GPUs   : {num_gpus}")
    print(f"  Llama GPU       : {llama_gpu}")

    captions_path = str(Path(output_path).with_suffix('.captions.json'))
    raw_path = str(Path(output_path).with_suffix('.scores.raw.jsonl'))
    require_new_paths(output_path, report_path, captions_path, raw_path)
    if llama_batch_size < 1:
        raise ValueError('llama-batch-size must be positive')
    set_seed(seed)
    if not 0 <= score_threshold <= 1:
        raise ValueError("score-threshold must be in [0, 1]")
    with open(input_path) as f:
        data: Dict[str, Any] = json.load(f)

    valid_entries = validate_annotations(data)
    print(f"  Loaded {len(valid_entries)} annotated videos.")

    # Phase 1: InternVL3.5-241B generates captions
    pair_captions = _phase1_generate_captions(valid_entries, video_dir, num_gpus, internvl_model, flash_attention)
    write_json(captions_path, pair_captions)

    # Phase 2: Llama-3.3-70B scores captions (InternVL already unloaded)
    pair_captions = _phase2_score_captions(pair_captions, llama_gpu, llama_batch_size, llama_model, num_gpus, raw_path)

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
        "invalid_pairs":   total_pairs - total_valid - sum(r["unvalidated_pairs"] for r in report.values()),
        "unvalidated_pairs": sum(r["unvalidated_pairs"] for r in report.values()),
        "retention_rate":  round(retention, 4),
        "mean_overall_score": round(mean_score, 4),
        "score_threshold": score_threshold,
        "internvl_model": internvl_model, "llama_model": llama_model, "seed": seed,
        "overall_rule": "arithmetic mean of plausibility, visual_evidence, strength, transition",
    }
    print(
        f"\n  Retained {total_valid}/{total_pairs} pairs "
        f"({100 * retention:.1f}%)  |  mean score: {mean_score:.3f}"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, {"aggregate": aggregate, "per_video": report, "pair_captions": pair_captions})
    if any(r["unvalidated_pairs"] for r in report.values()):
        raise RuntimeError("Unscored DCC pairs remain; see the report. Final labels were not written.")
    write_json(output_path, filtered)

    print(f"  Filtered annotations -> {output_path}")
    print(f"  Scoring report       -> {report_path}")


# CLI

def main() -> None:
    n_available = torch.cuda.device_count() if torch.cuda.is_available() else 1

    parser = argparse.ArgumentParser(
        description="DCC benchmark -- VLM+LLM causal pair verification (Step 3 of 4)"
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
        "--llama-gpu", type=int, default=None,
        help="Optional single GPU for Llama; default shards across --num-gpus after InternVL is freed",
    )
    parser.add_argument(
        "--llama-batch-size", type=int, default=1,
        help="Forward-pass batch size for Llama scoring (default: 1)",
    )
    parser.add_argument(
        "--score-threshold", type=float, required=True,
        help="Required mean-score cutoff in [0, 1]",
    )
    parser.add_argument('--internvl-model', default=INTERNVL_MODEL_ID, help='HF ID or local checkpoint directory')
    parser.add_argument('--llama-model', default=LLAMA_MODEL_ID, help='HF ID or local checkpoint directory')
    parser.add_argument('--flash-attention', action='store_true', help='Requires a compatible flash-attn installation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("score-threshold must be in [0, 1]")

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
        internvl_model = args.internvl_model, llama_model = args.llama_model,
        flash_attention = args.flash_attention, seed = args.seed,
    )


if __name__ == "__main__":
    main()
