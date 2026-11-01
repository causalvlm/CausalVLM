# Causal-VLM code

**[Published weights](https://huggingface.co/causalvlm/CausalVLM)** ·
**[DCC labels](https://huggingface.co/datasets/causalvlm/DCC)**

Run these entry points from the repository root:

| Command | Purpose |
| --- | --- |
| `python -m causal_vlm.prepare_video` | Prepare sampled frames and source timestamps |
| `python -m causal_vlm.train_stage1` | Train the captioning model |
| `python -m causal_vlm.train_stage2` | Train the causal head and joint model |
| `python -m causal_vlm.predict_published` | Run the published Hugging Face checkpoint |
| `python -m causal_vlm.predict` | Run a version-2 checkpoint saved by the training code |

See [data preparation](../docs/data.md), [training](../docs/training.md),
and [inference](../docs/inference.md) for complete commands and settings.
