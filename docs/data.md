# Data and downloads

## Published resources

- **Weights:** [causalvlm/CausalVLM](https://huggingface.co/causalvlm/CausalVLM)
- **DCC labels:** [causalvlm/DCC](https://huggingface.co/datasets/causalvlm/DCC)

Both resources require accepting their access terms on Hugging Face. Their model
and dataset cards specify CC BY-NC-ND 4.0. Pretrained backbones and source videos
have their own terms, available from their providers.

```bash
hf auth login
hf download causalvlm/DCC --repo-type dataset --local-dir data/dcc
```

| Dataset | Split | File inside `data/dcc/` |
| --- | --- | --- |
| ActivityNet | Train | `Activity_Net/AN_train_annotations_flat.jsonl` |
| ActivityNet | Validation | `Activity_Net/AN_val_annotations_flat.jsonl` |
| ActivityNet | Test | `Activity_Net/AN_test_annotations_flat.jsonl` |
| YouCook2 | Train | `Youcook2/YC2_train_annotations_flat.jsonl` |
| YouCook2 | Validation | `Youcook2/YC2_val_annotations_flat.jsonl` |

The labels contain annotations; obtain the matching videos from the original
ActivityNet and YouCook2 sources. Keep each released split separate.

## Convert labels for training

Each released JSONL row represents one event with these fields:

```text
video_id, duration, event_id, start, end, description, event_type,
is_cause, is_effect, cause_event, effect_event
```

Training preparation accepts a JSON object grouped by video. Convert the train
and validation files:

```bash
python dcc_benchmark/import_labels.py \
  --input data/dcc/Activity_Net/AN_train_annotations_flat.jsonl \
  --output data/annotations/train.json

python dcc_benchmark/import_labels.py \
  --input data/dcc/Activity_Net/AN_val_annotations_flat.jsonl \
  --output data/annotations/val.json
```

Conversion preserves event IDs, descriptions, timestamps, and causal links.
It validates complete graphs, consistent durations, and reciprocal edge references.
Invalid records stop conversion with an error; existing outputs are protected.
The inverse operation is [label export](../dcc_benchmark/README.md#export-the-released-label-format).

## Prepare videos

Create separate video manifests for training and validation. Each is a JSON
mapping from the exact DCC video ID to a local video path:

```json
{
  "video_001": "/path/to/videos/video_001.mp4",
  "video_002": "/path/to/videos/video_002.mp4"
}
```

Paths may also be relative to the manifest. Each manifest and its annotation file
must contain exactly the same video IDs. For ActivityNet, retain the label's
`v_` prefix in the manifest key even if the video filename omits it.

```bash
python -m causal_vlm.prepare_video \
  --videos videos_train.json --annotations data/annotations/train.json \
  --split train --output-dir data/frames_train

python -m causal_vlm.prepare_video \
  --videos videos_val.json --annotations data/annotations/val.json \
  --split val --output-dir data/frames_val
```

DINOv2 ViT-S/14 detects scene boundaries below cosine similarity 0.8. Scenes are
balanced to 64 by splitting at low-similarity boundaries or merging adjacent
scenes. One mid-frame represents each scene; videos shorter than 64 frames repeat
source frames. Decoded presentation timestamps and video duration are preserved.

```text
data/
  frames_train/
    frames/<video_id>/
    metadata/<video_id>.json
    annotations/train_processed.json
  frames_val/
    frames/<video_id>/
    metadata/<video_id>.json
    annotations/val_processed.json
```

Use `--data data` for training. Training rejects missing frames, inconsistent
metadata, and annotation intervals exceeding the decoded video duration. Resolve
source-video or timestamp mismatches before training; preprocessing does not
clip or rewrite the labels. Preparation requires a new output directory.

For inference, omit `--annotations`; see the [inference guide](inference.md).
