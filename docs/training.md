# Training CausalVLM

Install `causal_vlm/requirements.txt` and complete [data preparation](data.md).
The default recipe has two stages: captioning, followed by causal-head and joint
training. Run commands from the repository root.

## Architecture and initialization

| Component | Default |
| --- | --- |
| Perception encoder | `vit_pe_lang_large_patch14_448.fb` through timm; standalone [PE-Lang-L14-448](https://huggingface.co/facebook/PE-Lang-L14-448) weights |
| Language model | [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Visual-to-language projector | Linear → GELU → Linear → spatial average pooling |
| Projector initialization | Normal weights, standard deviation 0.02; zero biases |
| Frames | 64 at 448 × 448 |
| Spatial pooling ratio | 4; 32 × 32 patches become 8 × 8 tokens per frame |
| Causal head | Width 512; 8 attention heads; 2 transformer layers; FFN width 1024 |

The encoder and language model load from separate pretrained checkpoints. Stage 1
initializes and trains the projector. Stage 2 restores the complete Stage 1 model,
including the trained projector, then initializes the causal head.

Both [stage1.yaml](../causal_vlm/configs/stage1.yaml) and
[stage2.yaml](../causal_vlm/configs/stage2.yaml) use the same backbones and
preprocessing. The encoder outputs spatial patch tokens after removing one CLS
token. Projected features have the language-model width, 3072 by default.

`--encoder-checkpoint` and `--llm-checkpoint` accept model IDs or local directories.
Use `--encoder-loader timm` for timm names or `hf-hub:` references; the
`transformers` loader supports compatible standalone vision encoders. Match depth,
feature layer, patch grid, normalization, and prefix-token settings to the encoder.
Loading requires all backbone parameters and a Llama language-model configuration.

## Stage 1: captioning

Inside a four-GPU allocation:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module causal_vlm.train_stage1 \
  --config causal_vlm/configs/stage1.yaml \
  --data data --save-dir checkpoints/stage1
```

| Phase | Epochs | Trainable components | Peak learning rates | Warmup steps |
| --- | --- | --- | --- | --- |
| 1 | 5 | Projector | Projector: 1e-4 | 500 |
| 2 | 10 | Projector, encoder | Projector: 1e-4; encoder: 1e-5 | 500 |
| 3 | 10 | Projector, encoder, Llama | Projector: 5e-5; encoder: 1e-5; Llama: 1e-6 | 1000 |

## Stage 2: causal learning

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module causal_vlm.train_stage2 \
  --config causal_vlm/configs/stage2.yaml \
  --data data --stage1-ckpt checkpoints/stage1/last_model.pt \
  --save-dir checkpoints/stage2
```

| Phase | Epochs | Trainable components | Peak learning rates | Warmup steps |
| --- | --- | --- | --- | --- |
| 1 | 10 | Causal head | Head: 1e-4 | 500 |
| 2 | 10 | All components | Head: 1e-4; projector: 1e-5; encoder and Llama: 1e-6 | 500 |

The batch size of 16 is global across ranks. Stage 2 accumulates four batches,
for an effective batch size of 64. Both stages use AdamW, cosine decay after
warmup, and gradient checkpointing on the vision and language backbones.
The phase schedules are defined in [training.py](../causal_vlm/src/training.py).

The causal head pools projected frame features within annotated event intervals
and averages contextual final-layer Llama features over each event description.
Visual and text projections each use Linear → LayerNorm → GELU. Empty intervals
use the nearest sampled frame to their midpoint. Overlapping intervals may share
frames, and event counts remain dynamic.

The causal objective uses forward pairs, positive weight `n_negative/n_positive`,
and the hardest 20% of negatives (rounded down, with at least one when available).
Positive pairs below 0.5 receive a factor of 10. The reduction divides by all
eligible forward pairs. Graphs without positive edges remain training examples;
when all eligible pairs are positive, the positive-weight formula gives zero.
The joint loss is caption loss plus twice causal loss.

## Checkpoints and resource requirements

Version-2 `.pt` files contain model weights, configuration, architecture and
preprocessing signatures, optimizer/scheduler state, and epoch metadata.
`last_model.pt` is the latest checkpoint. `best_objective.pt` minimizes the
teacher-forced validation objective. Loading verifies tensor keys, shapes, and
the saved signature. `--init-checkpoint` initializes Stage 1 from saved weights
and starts a new curriculum; it does not resume optimizer state.

The default configuration requires four high-memory GPUs, with approximately
109 GiB of allocated memory per GPU. For smaller GPUs, reduce the global batch
size and adjust gradient accumulation.

For prediction, see [inference from trained checkpoints](inference.md#checkpoints-from-this-training-code).
