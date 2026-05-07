# Causal-VLM: Dense Causal Captioning in Videos

Training code for the Causal-VLM architecture for dense causal captioning in long-form videos.

---

## Repository Structure

```
causal_vlm/
├── train_stage1.py          # Stage 1: curriculum dense captioning
├── train_stage2.py          # Stage 2: multimodal causal head
├── configs/
│   ├── stage1.yaml
│   └── stage2.yaml
└── src/
    ├── models/
    │   ├── perception_lm.py      # Stage 1 backbone (Perception-L/14 + Llama-3.1-3B)
    │   └── causal_vlm_model.py   # Stage 2 causal head + CausalVLM wrapper
    ├── data/
    │   └── dataset.py            # DCC dataset loader and collate function
    └── utils/
        ├── checkpoint.py
        └── logger.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Model access — authenticate with HuggingFace before training:
```bash
huggingface-cli login
```

Models used:
- `facebook/Perception-LM-3B`
- `meta-llama/Llama-3.1-3B-Instruct`

---

## Dataset Format

Preprocessed frames and DCC annotations are expected at:

```
dataset_path/
├── frames_train/
│   ├── frames/              # clip frames as JPEG/PNG
│   └── annotations/
│       └── train_processed.json
└── frames_val/
    ├── frames/
    └── annotations/
        └── val_processed.json
```

The annotation JSON maps video IDs to events with timestamps, captions,
and causal adjacency structure. Use the `dcc_benchmark/` pipeline to generate these.

---

## Training

### Stage 1 — Dense Video Captioning

Edit `data.dataset_path` and `training.save_dir` in `configs/stage1.yaml`, then run:

```bash
python train_stage1.py \
    --config   configs/stage1.yaml \
    --exp-name stage1_v1
```

Three-phase curriculum (5 + 10 + 10 epochs):
- Phase 1: projection layer only
- Phase 2: projection + vision encoder
- Phase 3: full end-to-end

### Stage 2 — Multimodal Causal Head

Requires a Stage 1 checkpoint. Edit `configs/stage2.yaml` then run:

```bash
python train_stage2.py \
    --config      configs/stage2.yaml \
    --stage1-ckpt checkpoints/stage1/best_model.pt \
    --exp-name    stage2_v1
```

Two-phase training (10 + 10 epochs):
- Phase 1: causal head only, Stage 1 frozen
- Phase 2: full end-to-end

---

## Architecture

**Stage 1** (`src/models/perception_lm.py`)

Perception encoder-L/14 extracts per-frame CLS features.
A VisionProjector maps them to Llama's embedding space.
Llama-3.1-3B-Instruct (with LoRA) is trained with autoregressive caption loss.

**Stage 2** (`src/models/causal_vlm_model.py`)

Inputs:
- `V ∈ R^{B × F × D_v}` — visual features from Stage 1 encoder
- `T ∈ R^{B × N × D_t}` — event text embeddings from Stage 1 LLM token embedding layer

Pipeline: event-level pooling → cross-modal attention (V queries T) →
2-layer transformer encoder → pairwise MLP → causal adjacency matrix `A ∈ R^{N × N}`.

Joint loss: `L_total = L_caption + 2 * L_causal`

`L_causal` handles extreme class imbalance via:
- Uncapped positive weighting (`w_pos = n_neg / n_pos`)
- Hard negative mining (top-20% most-confusing negatives)
- Asymmetric false-negative penalty (`λ_fn = 10`)
