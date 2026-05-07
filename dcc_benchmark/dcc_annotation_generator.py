"""
DCC Benchmark -- Step 1: Causal Annotation Generation
======================================================
Generates causal adjacency annotations for the Dense Causal Captions (DCC)
benchmark by prompting Llama-3.3-70B with structured prompts over ActivityNet
and YouCook2 dense caption data.

Implements the annotation pipeline described in Section 3.2 and Appendix C
(LLM Annotation Generation Prompt) of the paper.

Pipeline
--------
    Step 1 -- dcc_annotation_generator.py  (this file)
              LLM causal annotation via Llama-3.3-70B
    Step 2 -- dcc_emscore_validation.py
              Tier-1 multimodal alignment via EMScore (threshold theta=0.2)
    Step 3 -- dcc_vlm_validation.py
              Tier-2 VLM+LLM causal-pair verification
              (InternVL3.5-241B caption generation + Llama-3.3-70B scoring)
    Step 4 -- human_study/
              Tier-3 human validation framework (ICC inter-annotator agreement)

Input format (per video)
------------------------
    {
        "video_id": {
            "duration": float,
            "timestamps": [[start, end], ...],
            "sentences": ["caption text", ...]
        }
    }

Output format (per video)
--------------------------
    {
        "video_id": {
            "duration": float,
            "events": [
                {
                    "event_id": int,
                    "timestamp": [start, end],
                    "description": "text",
                    "chain": {
                        "is_cause": bool,
                        "is_effect": bool,
                        "cause_event": [int, ...],
                        "effect_event": [int, ...]
                    },
                    "event_type": "cause|effect|causal|independent"
                }
            ]
        }
    }

Usage
-----
    python dcc_annotation_generator.py \\
        --input  train.json val_1.json val_2.json \\
        --output dcc_train.json dcc_val_1.json dcc_val_2.json \\
        --gpus   8 \\
        --batch-size 4
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import pipeline

# ---------------------------------------------------------------------------
# Model and prompt constants
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# System turn -- verbatim from Appendix C of the paper
_SYSTEM_PROMPT = (
    "You are an expert in analyzing video narratives and identifying "
    "cause-effect relationships in sequences of events."
)

# JSON schema shown verbatim in Appendix C of the paper
_JSON_SCHEMA = """{
    "duration": float,
    "events": [
        {
            "event_id": int,
            "timestamp": [start, end],
            "description": "text",
            "chain": {
                "is_cause": boolean,
                "is_effect": boolean,
                "cause_event": [event_ids],
                "effect_event": [event_ids]
            },
            "event_type": "cause,effect,causal,independent"
        }
    ]
}"""


def _build_user_prompt(video_data: Dict[str, Any]) -> str:
    """
    Constructs the user-turn prompt.

    Follows the LLM Annotation Generation Prompt in Appendix C of the paper
    verbatim, substituting {input_json} and {N} with the actual video data.
    """
    input_json = json.dumps(video_data, indent=2)
    n_events = len(video_data.get("timestamps", []))
    return (
        "Convert the following video caption data into a structured narrative "
        "format that shows cause-effect relationships between events.\n"
        "Input video caption data:\n"
        f"{input_json}\n\n"
        "Generate a single JSON object as output with exactly this structure:\n"
        f"{_JSON_SCHEMA}\n\n"
        "Guidelines for analysis:\n"
        "0. Count total number of events in input first\n"
        "1. Label each event as \"independent\", \"cause\", \"effect\" or \"causal\":\n"
        "   - \"cause\": Directly leads to or influences other events\n"
        "   - \"effect\": Results from or is influenced by other events\n"
        "   - \"causal\": Both causes and effects (two-way relationships)\n"
        "   - \"independent\": No causal relationships with other events\n"
        "2. Look for causal indicators\n"
        "3. Track relationships using event IDs\n"
        "4. Enforce temporal constraints:\n"
        "   - cause_event can ONLY reference prior event IDs\n"
        "   - effect_event can ONLY reference later event IDs\n"
        f"5. Do not add more than {n_events} events\n"
        "6. Output ONLY the JSON object, no additional text\n"
        "Provide only the JSON object output."
    )


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def _gpu_summary(gpu_id: int) -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    props = torch.cuda.get_device_properties(gpu_id)
    total = props.total_memory
    alloc = torch.cuda.memory_allocated(gpu_id)
    free  = total - torch.cuda.memory_reserved(gpu_id)

    def _fmt(b: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b //= 1024
        return f"{b:.1f} TB"

    return (
        f"GPU {gpu_id}: allocated={_fmt(alloc)}/{_fmt(total)}, "
        f"free={_fmt(free)}"
    )


def _clear_gpu(gpu_id: int) -> None:
    torch.cuda.set_device(gpu_id)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Causal annotator
# ---------------------------------------------------------------------------

class CausalAnnotator:
    """
    Wraps Llama-3.3-70B to generate DCC causal annotations for videos
    assigned to a single GPU.

    Parameters
    ----------
    gpu_id     : CUDA device index
    batch_size : number of videos per forward pass
    """

    def __init__(self, gpu_id: int, batch_size: int = 4) -> None:
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        device = f"cuda:{gpu_id}"

        torch.cuda.set_device(gpu_id)
        _clear_gpu(gpu_id)

        print(f"[GPU {gpu_id}] Loading {MODEL_ID} ...")
        t0 = time.time()
        self.pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            torch_dtype=torch.bfloat16,
            device=device,
            model_kwargs={"low_cpu_mem_usage": True},
            batch_size=batch_size,
        )
        # Left-padding is required for batched decoder-only generation.
        self.pipe.tokenizer.padding_side = "left"
        if self.pipe.tokenizer.pad_token is None:
            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token

        print(
            f"[GPU {gpu_id}] Model ready in {time.time() - t0:.1f}s  |  "
            f"{_gpu_summary(gpu_id)}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, video_data: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(video_data)},
        ]

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts the first complete JSON object from *text*.

        Uses raw_decode so that any text after the closing brace is ignored,
        which makes parsing robust to models that append explanations after
        the requested JSON object.
        """
        start = text.find("{")
        if start == -1:
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            return obj
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_batch(
        self,
        batch: List[Tuple[str, Dict[str, Any]]],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Annotate a batch of videos.

        Parameters
        ----------
        batch : list of (video_id, video_data) tuples

        Returns
        -------
        dict mapping video_id -> parsed annotation dict, or None on failure
        """
        if not batch:
            return {}

        video_ids   = [vid for vid, _ in batch]
        all_messages = [self._build_messages(data) for _, data in batch]

        t0 = time.time()
        responses = self.pipe(
            all_messages,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
            batch_size=self.batch_size,
        )
        print(
            f"[GPU {self.gpu_id}] Batch of {len(batch)} done in "
            f"{time.time() - t0:.1f}s"
        )

        results: Dict[str, Optional[Dict[str, Any]]] = {}
        for video_id, response in zip(video_ids, responses):
            text = response[0]["generated_text"]
            parsed = self._extract_json(text)
            if parsed is None:
                print(f"[GPU {self.gpu_id}] JSON extraction failed for {video_id}")
            results[video_id] = parsed

        del all_messages, responses
        gc.collect()
        return results


# ---------------------------------------------------------------------------
# Multi-GPU parallelism
# ---------------------------------------------------------------------------

def _split_data(
    videos: Dict[str, Any],
    n_parts: int,
) -> List[List[Tuple[str, Any]]]:
    """Divide the video dict into n_parts roughly equal chunks."""
    items = list(videos.items())
    chunk_size, remainder = divmod(len(items), n_parts)
    chunks, start = [], 0
    for i in range(n_parts):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def _process_chunk(args: Tuple[List, int, int]) -> Optional[str]:
    """
    Worker function executed in a subprocess for each GPU.

    Writes results to a temporary JSON file and returns its path so the
    main process can merge all GPU outputs.
    """
    chunk, gpu_id, batch_size = args
    torch.cuda.set_device(gpu_id)

    try:
        annotator = CausalAnnotator(gpu_id, batch_size)
        results: Dict[str, Optional[Dict]] = {}
        n_videos  = len(chunk)
        n_success = 0

        for start in tqdm(range(0, n_videos, batch_size), desc=f"GPU {gpu_id}"):
            batch = chunk[start: start + batch_size]
            batch_results = annotator.annotate_batch(batch)
            results.update(batch_results)
            n_success += sum(1 for v in batch_results.values() if v is not None)

        rate = 100 * n_success / n_videos if n_videos else 0.0
        print(f"[GPU {gpu_id}] Done: {n_success}/{n_videos} ({rate:.1f}%) successful")

        tmp = f"_dcc_tmp_{gpu_id}.json"
        with open(tmp, "w") as f:
            json.dump(results, f)
        return tmp

    except Exception as exc:
        print(f"[GPU {gpu_id}] Critical error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _verify_input(data: Dict[str, Any]) -> None:
    """Raise ValueError if any video entry is missing required fields."""
    required = {"duration", "timestamps", "sentences"}
    for vid, d in data.items():
        missing = required - set(d.keys())
        if missing:
            raise ValueError(f"Video {vid} missing fields: {missing}")
        if len(d["timestamps"]) != len(d["sentences"]):
            raise ValueError(
                f"Video {vid}: timestamps length ({len(d['timestamps'])}) "
                f"!= sentences length ({len(d['sentences'])})"
            )


# ---------------------------------------------------------------------------
# Dataset annotation entry point
# ---------------------------------------------------------------------------

def annotate_dataset(
    input_path: str,
    output_path: str,
    num_gpus: int,
    batch_size: int,
) -> None:
    """
    Annotate all videos in *input_path* and write results to *output_path*.

    Parameters
    ----------
    input_path  : path to input JSON file (ActivityNet or YouCook2 dense captions)
    output_path : path where the annotated JSON will be written
    num_gpus    : number of GPUs; one subprocess is spawned per GPU
    batch_size  : videos per forward pass per GPU
    """
    print(f"\n[annotate_dataset] {input_path} -> {output_path}")
    print(f"  GPUs={num_gpus}  batch_size={batch_size}")

    with open(input_path) as f:
        data: Dict[str, Any] = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{input_path} must be a dict mapping video_id -> video_data")

    _verify_input(data)
    print(f"  Loaded {len(data)} videos.")

    chunks = _split_data(data, num_gpus)
    worker_args = [
        (chunk, gpu_id, batch_size)
        for gpu_id, chunk in enumerate(chunks)
    ]

    t0 = time.time()
    with mp.Pool(num_gpus) as pool:
        tmp_files = pool.map(_process_chunk, worker_args)
    print(f"  Parallel annotation finished in {time.time() - t0:.1f}s")

    merged: Dict[str, Any] = {}
    for gpu_id, tmp in enumerate(tmp_files):
        if tmp and os.path.exists(tmp):
            with open(tmp) as f:
                partial = json.load(f)
            merged.update(partial)
            os.remove(tmp)
            print(f"  Merged {len(partial)} results from GPU {gpu_id}")
        else:
            print(f"  Warning: no results from GPU {gpu_id}")

    n_ok = sum(1 for v in merged.values() if v is not None)
    n_total = len(data)
    print(
        f"  Final: {n_ok}/{n_total} annotated "
        f"({100 * n_ok / n_total:.1f}%)"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  Saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _auto_batch_size() -> int:
    """Select batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 1
    gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if gb >= 70:   # H200 / A100 80 GB
        return 4
    if gb >= 40:   # A6000 48 GB
        return 2
    return 1       # smaller GPUs


def main() -> None:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="DCC benchmark -- causal annotation generation (Step 1 of 3)"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        metavar="FILE",
        help="Input JSON file(s) containing dense captions",
    )
    parser.add_argument(
        "--output", nargs="+", required=True,
        metavar="FILE",
        help="Output JSON file(s) for causal annotations (same count as --input)",
    )
    parser.add_argument(
        "--gpus", type=int,
        default=min(8, torch.cuda.device_count()) if torch.cuda.is_available() else 1,
        help="Number of GPUs to use (default: all available, up to 8)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Videos per forward pass per GPU (auto-detected from GPU memory if omitted)",
    )
    args = parser.parse_args()

    if len(args.input) != len(args.output):
        parser.error("--input and --output must specify the same number of files")

    num_gpus   = min(args.gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    batch_size = args.batch_size if args.batch_size is not None else _auto_batch_size()

    print(f"Model      : {MODEL_ID}")
    print(f"GPUs       : {num_gpus}")
    print(f"Batch size : {batch_size}")

    for inp, out in zip(args.input, args.output):
        annotate_dataset(inp, out, num_gpus, batch_size)


if __name__ == "__main__":
    main()
