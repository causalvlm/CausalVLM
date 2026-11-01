# DCC benchmark construction

**Released labels: [causalvlm/DCC on Hugging Face](https://huggingface.co/datasets/causalvlm/DCC)**

Generate causal annotations from human captions, validate proposed pairs against
video, prepare human-rating tasks, and export the final labels.
For existing labels and training preparation, see [data and downloads](../docs/data.md).

## Setup

Run commands from the repository root. Install ffmpeg and the Python dependencies:

```bash
python -m pip install -r dcc_benchmark/requirements.txt
```

Tier 1 uses the upstream EMScore implementation at a fixed revision. It has no
Python package installer; clone it into a separate dependency directory and add
that directory to `PYTHONPATH`:

```bash
git clone https://github.com/ShiYaya/EMScore.git /path/to/dependencies/EMScore
git -C /path/to/dependencies/EMScore checkout 20bce0ae234b3caeea3357c023c4d085c9a55bf0
export PYTHONPATH=/path/to/dependencies/EMScore${PYTHONPATH:+:$PYTHONPATH}
```

`emscore_backend.py` supplies the token-feature extension documented by EMScore,
using an in-memory CLIP adapter. It reuses the loaded ViT-B/32 weights across
pairs and leaves installed CLIP files unchanged. No separate CLIP fork is needed.

Llama-3.3-70B-Instruct and InternVL3.5-241B-A28B need substantial GPU memory.
The loaders shard one model across the allocated GPUs and unload InternVL before
loading the scoring Llama. InternVL's memory budget follows actual free GPU
memory. Flash attention is optional (`--flash-attention`) and requires a compatible
installation. Local checkpoint paths are accepted, including for offline runs.

## 1. Generate annotations

```bash
python dcc_benchmark/prepare_inputs.py \
  --input youcook2_annotations.json --split training --output yc2_input.json

python dcc_benchmark/dcc_annotation_generator.py \
  --input yc2_input.json --output dcc_yc2_raw.json \
  --model /models/Llama-3.3-70B-Instruct --batch-size 1
```

ActivityNet split JSONs containing `duration`, `timestamps`, and `sentences` are
accepted directly. YouCook2's `database`/`annotations` structure is converted
explicitly. Input events are stably ordered by onset, including overlaps.
`--dry-run` saves prompts without loading a model; `--max-videos` limits the number of videos processed.

The annotation prompt is the reference prompt shown in the paper's appendix,
versioned `dcc-appendix-reference-v1`. Structural validation runs after generation:
exactly N events, unchanged human captions and intervals, sequential IDs, forward
edges, reciprocal references, and consistent event roles. An edge-free graph is
valid. Validation does not infer or remove semantically redundant edges.

Raw responses are saved before validation. The accompanying manifest records the
model, seed, input hash, prompt version, and failures. Invalid responses produce
explicitly partial annotations and no final label file.

## 2–3. Validate proposed pairs

```bash
python dcc_benchmark/dcc_emscore_validation.py \
  --input dcc_yc2_raw.json --video-dir /data/youcook2/videos \
  --output dcc_yc2_t1.json

python dcc_benchmark/dcc_vlm_validation.py \
  --input dcc_yc2_t1.json --video-dir /data/youcook2/videos \
  --output dcc_yc2_t2.json --num-gpus 4 \
  --internvl-model /models/InternVL3_5-241B-A28B \
  --llama-model /models/Llama-3.3-70B-Instruct \
  --score-threshold YOUR_EXPERIMENT_CUTOFF
```

For each proposed edge, its two segments are concatenated in onset order.
Tier 1 retains pairs with video-grounded `EMScore(X,V).full_F >= 0.2` and uses no
text reference. Tier 2 captions eight uniformly sampled frames with InternVL,
then applies the appendix's Llama scoring prompt. The retention score is computed
as the arithmetic mean of plausibility, visual evidence, strength, and transition.
The model-reported overall value is retained for inspection, but does not override
that mean. Every component must be a finite number in [0, 1]. Set the Tier-2 retention cutoff explicitly for each construction run; it has no default.

Source videos may be flat or nested. Both `v_ID` and `ID` filenames are supported.
Input validation rejects malformed annotations, null records, and inconsistent chains. Missing videos, failed clips, or unavailable
scores block final labels. Pair reports record incomplete validation separately
from rejected edges. Tier 2 also saves captions before scoring and raw Llama
responses as scoring progresses. Use new output/report paths for every run.

## 4. Prepare and apply human ratings

```bash
python dcc_benchmark/dcc_human_study_tasks.py \
  --activitynet dcc_an_t2.json --youcook2 dcc_yc2_t2.json \
  --actnet-video-dir /data/activitynet/videos --yc2-video-dir /data/youcook2/videos \
  --high-event-weight YOUR_ALLOCATION_WEIGHT --output-dir pair_study
```

The defaults select 1,200 ActivityNet and 800 YouCook2 videos containing proposed
pairs, with three ratings per pair. `--activitynet-sample` and `--youcook2-sample`
set the sample sizes. Insufficient eligible videos are an error.

Videos with at least five events receive an explicit allocation weight greater
than 1. With H high-event videos and L low-event videos, the high quota is
`ceil(n * weight * H / (weight * H + L))`, capped by stratum availability.
This implements oversampling relative to the eligible population, including when
most videos already have five or more events. Set the allocation weight explicitly for each run. The selection manifest records
the weight, seed, exact selected IDs, and video counts for both strata.

Every proposed pair in the selected videos must have a usable clip. Failures
produce `tasks.partial.json` and no ready task list or rating template. Successful
runs produce `tasks.json`, `annotations_template.csv`, and annotator instructions.
Existing task directories are refused so completed ratings survive reruns.

```bash
python dcc_benchmark/dcc_apply_human_ratings.py \
  --input dcc_an_t2.json --tasks pair_study/tasks.json \
  --ratings pair_study/completed.csv --dataset activitynet \
  --threshold YOUR_HUMAN_CUTOFF --output dcc_an_final.json
```

All three integer ratings (1–5) are required for every sampled pair. Retention uses
their mean and the explicit cutoff. Empty tasks, missing clips, incomplete pair
coverage, and pairs absent from the input block filtering. Unsampled videos keep
their Tier-2 labels. Chain lists and event roles are rebuilt after edge removal.

## Export the released label format

```bash
python dcc_benchmark/export_labels.py \
  --input dcc_an_final.json --output AN_train_annotations_flat.jsonl
```

Each JSONL row contains `video_id`, `duration`, `event_id`, `start`, `end`,
`description`, `event_type`, `is_cause`, `is_effect`, `cause_event`, and
`effect_event`, matching the released DCC format. Export preserves timestamps,
IDs, captions, and relationships. It neither modifies existing label files nor
changes dataset split membership.

## Blind task preparation

```bash
python dcc_benchmark/dcc_blind_audit_tasks.py \
  --activitynet an_input.json --youcook2 yc2_input.json --output-dir blind_tasks
```

This optional task-preparation command uses original human-caption inputs,
selects 100 videos per dataset stratified proportionally by event count, and
lists every forward pair. It supplies blank labels and full-video instructions
without exposing LLM labels. Allocation uses largest remainders. The resulting tasks are ready for independent human annotation.
