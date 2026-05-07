# DCC Benchmark: Dense Causal Captions Dataset Construction Pipeline

This repository contains the annotation and validation pipeline for the
**Dense Causal Captions (DCC)** benchmark introduced in the paper:

> **Causal-VLM: Dense Causal Captioning in Videos**

DCC is the first large-scale benchmark for dense causal captioning, comprising
**21,586 videos** and **85,187 events** across ActivityNet Captions and YouCook2,
annotated with causal adjacency matrices and validated through a three-tier
pipeline.

---

## Pipeline Overview

The benchmark is constructed in four sequential steps.
Each step's output is the next step's input.

```
Raw dense captions (ActivityNet / YouCook2)
              │
              ▼
  ┌─────────────────────────────────────────┐
  │  Step 1 · dcc_annotation_generator.py  │
  │  Llama-3.3-70B structured prompting    │
  │  → causal adjacency annotations        │
  └─────────────────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────────────────┐
  │  Step 2 · dcc_emscore_validation.py    │
  │  Tier-1: EMScore multimodal alignment  │
  │  threshold θ = 0.2                     │
  │  → 94.3 % / 91.8 % pairs retained     │
  └─────────────────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────────────────┐
  │  Step 3 · dcc_vlm_validation.py        │
  │  Tier-2: InternVL3.5-241B captions     │
  │          + Llama-3.3-70B scoring       │
  │  → mean score 0.628 AN / 0.632 YC2     │
  └─────────────────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────────────────┐
  │  Step 4a · dcc_human_study_tasks.py    │
  │  Stratified sampling (1,200 AN +        │
  │  800 YC2 videos, oversample 5+ events) │
  │  + merged clip generation for raters   │
  └─────────────────────────────────────────┘
              │   ← human annotators fill in ratings
              ▼
  ┌─────────────────────────────────────────┐
  │  Step 4b · dcc_human_study_score.py    │
  │  Tier-3: ICC(2,1) = 0.85               │
  │          (95 % CI: 0.83–0.91)          │
  │  → 98.5 % pairs score ≥ 3             │
  └─────────────────────────────────────────┘
              │
              ▼
     Final DCC Benchmark JSON
```

---

## Benchmark Statistics

| Dataset | Videos | Events | Causal pairs | Mean events/video |
|---|---|---|---|---|
| ActivityNet | 19,796 | 71,532 | 57,813 | 3.61 |
| YouCook2 | 1,790 | 13,812 | 16,375 | 7.72 |
| **Total** | **21,586** | **85,344** | **74,188** | — |

Causal chain lengths (YouCook2): 75.2 % of videos contain chains of four or
more hops, reflecting the dense procedural structure of cooking videos.

---

## Repository Structure

```
dcc_benchmark/
├── README.md
├── requirements.txt
├── dcc_annotation_generator.py   # Step 1 — LLM causal annotation
├── dcc_emscore_validation.py     # Step 2 — Tier-1 EMScore validation
├── dcc_vlm_validation.py         # Step 3 — Tier-2 VLM+LLM validation
├── dcc_human_study_tasks.py      # Step 4a — Human study task generation
└── dcc_human_study_score.py      # Step 4b — ICC scoring and final filtering
```

---

## Requirements

### Python packages

```bash
pip install -r requirements.txt
```

### System dependency

ffmpeg is required for all video segment extraction steps:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Model access

Both models must be accessible via the HuggingFace Hub.
Obtain access at:

- `meta-llama/Llama-3.3-70B-Instruct` — [https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- `OpenGVLab/InternVL3_5-241B-A28B` — [https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B)

```bash
huggingface-cli login
```

### GPU requirements

| Step | Model | Minimum GPU |
|---|---|---|
| 1 | Llama-3.3-70B | 1 × H200 (or 2 × A100 80 GB) |
| 2 | EMScorer (CLIP-based) | 1 × GPU with ≥ 16 GB |
| 3 (InternVL) | InternVL3.5-241B | ≥ 4 × H200 |
| 3 (Llama) | Llama-3.3-70B | 1 × H200 |
| 4 | — | CPU sufficient |

---

## Dataset Preparation

Download ActivityNet Captions and YouCook2 dense caption splits and place the
JSON files in a working directory.

Expected input format for Steps 1–2 (one file per split):

```json
{
  "video_id": {
    "duration": 120.5,
    "timestamps": [[0.0, 12.3], [12.3, 45.1], ...],
    "sentences": ["caption for event 1", "caption for event 2", ...]
  }
}
```

This matches the standard ActivityNet Captions and YouCook2 annotation format.

---

## Usage

### Step 1 — Causal Annotation Generation

Prompts Llama-3.3-70B to annotate causal relationships from dense captions.
Runs in parallel across multiple GPUs (one subprocess per GPU).

```bash
python dcc_annotation_generator.py \
    --input  train.json val_1.json val_2.json \
    --output dcc_train.json dcc_val_1.json dcc_val_2.json \
    --gpus   8 \
    --batch-size 4
```

| Argument | Description |
|---|---|
| `--input` | One or more input JSON files |
| `--output` | Corresponding output JSON files (same count) |
| `--gpus` | Number of GPUs (default: all available, up to 8) |
| `--batch-size` | Videos per forward pass per GPU (auto-detected if omitted) |

---

### Step 2 — EMScore Multimodal Alignment Validation (Tier 1)

For each causal pair, merges video segments and computes EMScore(X,V).
Pairs below θ = 0.2 are removed (Section 3.2 of paper).

```bash
# ActivityNet
python dcc_emscore_validation.py \
    --input    dcc_train.json \
    --video-dir /data/activitynet/videos \
    --output   dcc_train_t1.json

# YouCook2 (nested video subdirectories resolved automatically)
python dcc_emscore_validation.py \
    --input    dcc_yc2_train.json \
    --video-dir /data/youcook2/videos \
    --output   dcc_yc2_train_t1.json
```

| Argument | Description |
|---|---|
| `--input` | Step-1 annotation JSON |
| `--video-dir` | Root video directory (flat or nested layout) |
| `--output` | Filtered annotation JSON |
| `--report` | Per-pair score report JSON (default: `<output>_report.json`) |

---

### Step 3 — VLM+LLM Causal Pair Verification (Tier 2)

Two-phase: InternVL3.5-241B captions each merged clip, then Llama-3.3-70B
scores four dimensions (plausibility, visual evidence, strength, transition)
using the prompt in Appendix C of the paper.
InternVL is unloaded before Llama is initialised.

```bash
python dcc_vlm_validation.py \
    --input      dcc_train_t1.json \
    --video-dir  /data/activitynet/videos \
    --output     dcc_train_t2.json \
    --num-gpus   8 \
    --llama-gpu  0
```

| Argument | Description |
|---|---|
| `--input` | Step-2 filtered annotation JSON |
| `--video-dir` | Root video directory |
| `--output` | Tier-2-filtered annotation JSON |
| `--num-gpus` | GPUs for InternVL3.5-241B (default: all available) |
| `--llama-gpu` | GPU index for Llama after InternVL is freed (default: 0) |
| `--score-threshold` | Llama overall score cutoff (default: 0.5) |
| `--report` | Per-pair scoring report JSON |

---

### Step 4a — Human Study Task Generation (Tier 3)

Stratified sample of 2,000 videos (1,200 ActivityNet + 800 YouCook2),
oversampling videos with ≥ 5 events. Creates merged clips and an annotation
CSV template for three annotators.

```bash
python dcc_human_study_tasks.py \
    --activitynet      dcc_actnet_t2.json \
    --youcook2         dcc_yc2_t2.json \
    --actnet-video-dir /data/activitynet/videos \
    --yc2-video-dir    /data/youcook2/videos \
    --output-dir       human_study/
```

Outputs inside `human_study/`:

| File | Description |
|---|---|
| `tasks.json` | Full task manifest (one entry per causal pair) |
| `annotations_template.csv` | Blank CSV for annotators to complete |
| `clips/` | One merged `.mp4` per causal pair |
| `sampling_stats.json` | Stratum breakdown and sampling parameters |

Annotators watch each clip and enter a rating (1–5) for columns
`rater_1`, `rater_2`, `rater_3` in the template CSV.

---

### Step 4b — Human Annotation Scoring and ICC

Aggregates three ratings per pair, computes ICC(2,1) with 95 % CI, filters
pairs with mean rating < 3.0, and optionally writes the final DCC JSONs.

```bash
python dcc_human_study_score.py \
    --tasks        human_study/tasks.json \
    --annotations  human_study/annotations_completed.csv \
    --activitynet  dcc_actnet_t2.json \
    --youcook2     dcc_yc2_t2.json \
    --output-dir   human_study/results/
```

| Argument | Description |
|---|---|
| `--tasks` | `tasks.json` from Step 4a |
| `--annotations` | Completed annotation CSV |
| `--activitynet` | Step-3 ActivityNet JSON (optional; applies final filter) |
| `--youcook2` | Step-3 YouCook2 JSON (optional; applies final filter) |
| `--output-dir` | Directory for all output files |
| `--threshold` | Mean rating threshold for retention (default: 3.0) |
| `--alpha` | Significance level for ICC CI (default: 0.05) |

Outputs inside `results/`:

| File | Description |
|---|---|
| `human_validation_report.json` | ICC(2,1), CI, aggregate and per-dataset stats |
| `pair_results.json` | Per-pair mean rating and retention decision |
| `pairs_retained.csv` / `pairs_rejected.csv` | Audit trail |
| `dcc_actnet_final.json` | Final ActivityNet DCC annotations |
| `dcc_youcook2_final.json` | Final YouCook2 DCC annotations |

---

## Annotation Format

All intermediate and final JSON files share the same schema:

```json
{
  "video_id": {
    "duration": 120.5,
    "events": [
      {
        "event_id": 1,
        "timestamp": [0.0, 12.3],
        "description": "person seasons the chicken",
        "chain": {
          "is_cause": true,
          "is_effect": false,
          "cause_event": [],
          "effect_event": [3]
        },
        "event_type": "cause"
      }
    ]
  }
}
```

`event_type` takes one of four values matching the paper's taxonomy:

| Value | Meaning |
|---|---|
| `"cause"` | Initiates subsequent events only |
| `"effect"` | Results from prior events only |
| `"causal"` | Both causes and is caused (bidirectional mediator) |
| `"independent"` | No causal links with any other event |

---

## Validation Prompts

The exact prompts used for Steps 1 and 3 are reproduced in Appendix C of the
paper and are implemented verbatim in:

- **Step 1** — `_build_user_prompt()` in `dcc_annotation_generator.py`
- **Step 3** — `_build_scoring_prompt()` in `dcc_vlm_validation.py`

---

