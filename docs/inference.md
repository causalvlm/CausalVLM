# Inference

**Model download: [causalvlm/CausalVLM on Hugging Face](https://huggingface.co/causalvlm/CausalVLM)**

Install the dependencies with `python -m pip install -r causal_vlm/requirements.txt`
and run commands from the repository root.

| Checkpoint | Entry point | Input |
| --- | --- | --- |
| Published Hugging Face weights | `causal_vlm.predict_published` | Video file(s) |
| Version-2 `.pt` saved by this training code | `causal_vlm.predict` | Prepared frames and video metadata |

## Published Hugging Face weights

Accept the model's access terms on Hugging Face, then authenticate:

```bash
hf auth login
python -m causal_vlm.predict_published \
  --checkpoint causalvlm/CausalVLM \
  --video /path/to/video.mp4 \
  --output outputs/predictions.json
```

To use an existing download, replace the model ID with its local directory.
Multiple videos can follow `--video`. Each video's results include event onsets
and descriptions, an adjacency matrix, and causal edges as pairs of one-based
event IDs. Existing outputs are protected.

Default settings:

| Setting | Value |
| --- | --- |
| Frame sampling | 64 uniform frames |
| Input resolution | 448 × 448 |
| Projector pooling ratio | 4 |
| Event representation | Generated onset and description |
| Causal-head features | Uniform visual event pooling and mean token embeddings |
| Maximum generated tokens | 256 (`--max-new-tokens`) |
| Edge retention | Probability ≥ 0.4 (`--edge-threshold`); increasing event IDs only |

Onsets and causal links are model predictions. The published parser retains raw
onsets, which can be out of order or outside the video duration; increasing event
IDs alone do not guarantee valid temporal ordering. No end times are inferred in
this entry point. Check timestamp validity before downstream temporal use.

## Checkpoints from this training code

Create a video manifest as described in [data preparation](data.md), then prepare
frames without annotations:

```bash
python -m causal_vlm.prepare_video \
  --videos new_videos.json --output-dir data/inference_frames

python -m causal_vlm.predict \
  --checkpoint checkpoints/stage2/last_model.pt \
  --input data/inference_frames/inference_inputs.json \
  --output outputs/trained_predictions.json
```

Input records contain `frame_paths`, `frame_timestamps`, and `duration`. Frame
paths are relative to the input JSON. Model and preprocessing settings are
restored from the checkpoint. If backbone downloads moved, pass
`--encoder-checkpoint` and `--llm-checkpoint` with their new locations.

Outputs contain generated text, event intervals, a probability matrix, and edges
above 0.5 by default (`--edge-threshold`). Stage-1 checkpoints return captions and
events only. Onset-only captions are supported using the next strictly later
onset or the video's duration for the interval end. Malformed events are omitted;
an empty event list retains the raw generated text.

This path uses scene-aware frames, timestamp-based visual pooling, and contextual
Llama event features saved by the training configuration. It requires version-2
checkpoints; the published Hugging Face export uses the separate command above.
