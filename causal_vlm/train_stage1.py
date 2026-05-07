"""
Stage 1: Dense video captioning with three-phase curriculum training.

Phase 1 (5 ep)  — projection only.          LR: proj=1e-4
Phase 2 (10 ep) — projection + vision enc.  LR: proj=1e-4, vis=1e-5
Phase 3 (10 ep) — full end-to-end.          LR: proj=5e-5, vis=1e-5, llm=1e-6

All phases: AdamW (β1=0.9, β2=0.999, ε=1e-8), cosine LR, grad clip=1.0, bfloat16

Usage:
    python train_stage1.py --config configs/stage1.yaml --exp-name stage1_v1
"""

import argparse
import math

import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageTextToText

from src.data.dataset import SimplePreprocessedCausalVideoDataset, simple_collate_fn
from src.models.perception_lm import PerceptionVideoLM
from src.utils.checkpoint import load_checkpoint, save_checkpoint

PHASE_EPOCHS  = [5, 10, 10]
WARMUP_STEPS  = [500, 500, 1000]
TOTAL_EPOCHS  = sum(PHASE_EPOCHS)
BOUNDARIES    = [PHASE_EPOCHS[0], PHASE_EPOCHS[0] + PHASE_EPOCHS[1], TOTAL_EPOCHS]


def current_phase(epoch: int) -> int:
    if epoch < BOUNDARIES[0]:
        return 1
    if epoch < BOUNDARIES[1]:
        return 2
    return 3


def build_optimizer(model: PerceptionVideoLM, phase: int) -> torch.optim.Optimizer:
    for p in model.parameters():
        p.requires_grad = False

    enc  = model.encoder.vision_model
    proj = model.decoder.visual_proj
    llm  = model.decoder.llama

    if phase == 1:
        for p in proj.parameters():
            p.requires_grad = True
        groups = [{"params": list(proj.parameters()), "lr": 1e-4}]

    elif phase == 2:
        for m in (proj, enc):
            for p in m.parameters():
                p.requires_grad = True
        groups = [
            {"params": list(proj.parameters()), "lr": 1e-4},
            {"params": list(enc.parameters()),  "lr": 1e-5},
        ]

    else:
        for p in model.parameters():
            p.requires_grad = True
        groups = [
            {"params": list(proj.parameters()), "lr": 5e-5},
            {"params": list(enc.parameters()),  "lr": 1e-5},
            {"params": list(llm.parameters()),  "lr": 1e-6, "weight_decay": 0.01},
        ]

    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(optimizer, n_steps: int, warmup: int):
    def lr_fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def train_epoch(model, loader, optimizer, scheduler, device, config, epoch):
    model.train()
    tok       = model.decoder.tokenizer
    acc       = config.training.gradient_accumulation_steps
    total     = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc=f"Ep {epoch+1} train")):
        frames = batch["frames"].to(device, dtype=torch.bfloat16)
        ids    = tok(
            batch["captions"], return_tensors="pt", padding=True,
            truncation=True, max_length=config.data.max_caption_length,
        ).to(device)

        loss = model(frames, labels=ids.input_ids).loss / acc
        if torch.isfinite(loss):
            loss.backward()
            total += loss.item() * acc

        if (step + 1) % acc == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.training.clip_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if wandb.run:
                wandb.log({"train/loss": loss.item() * acc,
                           "train/lr": scheduler.get_last_lr()[0], "epoch": epoch})

    return total / len(loader)


def val_epoch(model, loader, device, config, epoch):
    model.eval()
    tok   = model.decoder.tokenizer
    total = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Ep {epoch+1} val"):
            frames = batch["frames"].to(device, dtype=torch.bfloat16)
            ids    = tok(
                batch["captions"], return_tensors="pt", padding=True,
                truncation=True, max_length=config.data.max_caption_length,
            ).to(device)
            total += model(frames, labels=ids.input_ids).loss.item()
    avg = total / len(loader)
    if wandb.run:
        wandb.log({"val/loss": avg, "epoch": epoch})
    return avg


def main(args):
    config = OmegaConf.load(args.config)
    wandb.init(project=config.logging.project_name, name=args.exp_name,
               config=OmegaConf.to_container(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading vision tower ...")
    plm = AutoModelForImageTextToText.from_pretrained(
        "facebook/Perception-LM-3B", torch_dtype=torch.bfloat16
    )
    vt = plm.model.vision_tower
    for attr in ("hidden_size", "num_features", "embed_dim"):
        if hasattr(vt.config, attr):
            config.model.encoder.hidden_size = getattr(vt.config, attr)
            break
    del plm

    model = PerceptionVideoLM(config, vision_model=vt).to(device, dtype=torch.bfloat16)
    if args.resume:
        load_checkpoint(args.resume, model)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_ds = SimplePreprocessedCausalVideoDataset(config.data.dataset_path, "train", 448)
    val_ds   = SimplePreprocessedCausalVideoDataset(config.data.dataset_path, "val",   448)
    kw       = dict(collate_fn=simple_collate_fn, num_workers=config.data.num_workers,
                    pin_memory=True)
    train_loader = DataLoader(train_ds, config.data.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   config.data.batch_size, shuffle=False, **kw)

    best_loss  = float("inf")
    prev_phase = None

    for epoch in range(TOTAL_EPOCHS):
        phase = current_phase(epoch)
        if phase != prev_phase:
            actual = model.module if hasattr(model, "module") else model
            opt    = build_optimizer(actual, phase)
            n_steps = PHASE_EPOCHS[phase - 1] * len(train_loader)
            sched  = build_scheduler(opt, n_steps, WARMUP_STEPS[phase - 1])
            prev_phase = phase
            print(f"\n--- Phase {phase} ---")

        train_loss = train_epoch(model, train_loader, opt, sched, device, config, epoch)
        val_loss   = val_epoch(model, val_loader, device, config, epoch)
        print(f"Ep {epoch+1:3d} | phase={phase} | train={train_loss:.4f} | val={val_loss:.4f}")

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        actual = model.module if hasattr(model, "module") else model
        save_checkpoint(
            {"epoch": epoch + 1, "phase": phase,
             "model_state_dict": actual.state_dict(),
             "best_val_loss": best_loss,
             "config": OmegaConf.to_container(config)},
            is_best, config.training.save_dir, f"stage1_ep{epoch+1:03d}.pt",
        )

    wandb.finish()
    print(f"Stage 1 done. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--exp-name",  default="stage1")
    p.add_argument("--resume",    default=None)
    main(p.parse_args())
