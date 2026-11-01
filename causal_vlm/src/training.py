"""Training curricula with teacher-forced validation of the training objective.
"""
from __future__ import annotations
import contextlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .data.dataset import SimplePreprocessedCausalVideoDataset, simple_collate_fn
from .losses import causal_loss

STAGE1_PHASES = [
    {"epochs": 5, "warmup": 500, "projector": 1e-4},
    {"epochs": 10, "warmup": 500, "projector": 1e-4, "vision": 1e-5},
    {"epochs": 10, "warmup": 1000, "projector": 5e-5, "vision": 1e-5, "language": 1e-6},
]
STAGE2_PHASES = [
    {"epochs": 10, "warmup": 500, "head": 1e-4},
    {"epochs": 10, "warmup": 500, "head": 1e-4, "projector": 1e-5,
     "vision": 1e-6, "language": 1e-6},
]


def phase_at(epoch, phases):
    start = 0
    for index, phase in enumerate(phases):
        if start <= epoch < start + phase["epochs"]:
            return index
        start += phase["epochs"]
    raise ValueError("Epoch is outside the curriculum")


def build_optimizer(model, phase, stage):
    captioner = model if stage == 1 else model.stage1
    modules = {"projector": captioner.decoder.visual_proj,
               "vision": captioner.encoder.vision_model, "language": captioner.decoder.llama}
    if stage == 2:
        modules["head"] = model.causal_head
    model.requires_grad_(False)
    groups = []
    for name, module in modules.items():
        if name in phase:
            module.requires_grad_(True)
            groups.append({"params": list(module.parameters()), "lr": phase[name],
                           "weight_decay": 0.01, "name": name})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(optimizer, steps, warmup):
    if steps < 1 or warmup < 0:
        raise ValueError("Invalid scheduler steps or warmup")
    def factor(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = min(1.0, (step-warmup) / max(1, steps-warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def caption_tokens(tokenizer, captions, max_length, device):
    if max_length < 2 or tokenizer.eos_token_id is None:
        raise ValueError("Caption training requires EOS and max_length >= 2")
    previous = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        tokens = tokenizer([c + tokenizer.eos_token for c in captions], padding=True,
                           truncation=True, max_length=max_length, return_tensors="pt",
                           add_special_tokens=False)
    finally:
        tokenizer.padding_side = previous
    ids, mask = tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
    # Keep an explicit EOS even when a long caption is truncated.
    ids[torch.arange(len(ids), device=device), mask.sum(1)-1] = tokenizer.eos_token_id
    return ids, mask


def batch_objective(model, batch, config, device, stage):
    core = model.module if isinstance(model, DistributedDataParallel) else model
    captioner = core if stage == 1 else core.stage1
    ids, mask = caption_tokens(captioner.decoder.tokenizer, batch["captions"],
                               config["data"]["max_caption_length"], device)
    frames = batch["frames"].to(device)
    if stage == 1:
        cap = model(frames, labels=ids, label_attention_mask=mask).loss
        return cap, {"caption": cap, "causal": cap.detach() * 0}
    out = model(frames, batch["event_descriptions"], batch["num_events"], labels=ids,
                label_attention_mask=mask, frame_timestamps=batch["frame_timestamps"].to(device),
                event_timestamps=batch["event_timestamps"].to(device))
    cfg = config["loss"]
    losses = [causal_loss(out["adjacency_matrix"][b, :n, :n],
                          batch["adjacency_matrix"][b, :n, :n].to(device),
                          cfg["hard_negative_ratio"], cfg["false_negative_weight"])
              for b, n in enumerate(batch["num_events"])]
    caus = torch.stack(losses).mean()
    cap = out["caption_loss"]
    return cap + cfg["causal_weight"] * caus, {"caption": cap, "causal": caus}


def run_epoch(model, loader, optimizer, scheduler, device, config, stage, training,
              objective=batch_objective):
    model.train(training)
    core = model.module if isinstance(model, DistributedDataParallel) else model
    captioner = core if stage == 1 else core.stage1
    if training:
        # Frozen pretrained modules must not inject training-time dropout.
        for module in (captioner.encoder.vision_model, captioner.decoder.llama):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
    steps = len(loader)
    acc = config["training"]["gradient_accumulation_steps"] if training else 1
    if acc < 1:
        raise ValueError("Gradient accumulation must be positive")
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    if training:
        optimizer.zero_grad(set_to_none=True)
    updates = 0
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        window_start = (step // acc) * acc
        window_size = min(acc, steps-window_start)
        finish = (step+1) % acc == 0 or step+1 == steps
        sync = model.no_sync() if training and hasattr(model, "no_sync") and not finish else contextlib.nullcontext()
        with sync, torch.set_grad_enabled(training):
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                loss, parts = objective(model, batch, config, device, stage)
            finite = torch.isfinite(loss).all().to(dtype=torch.int32)
            if training and dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                raise FloatingPointError("Nonfinite training objective; stopped without skipping samples")
            if training:
                (loss / window_size).backward()
        count = len(batch["frames"])
        totals[:3] += torch.stack([loss.detach(), parts["caption"].detach(), parts["causal"].detach()]).double() * count
        totals[3] += count
        if training and finish:
            torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad],
                                           config["training"]["clip_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
    if dist.is_initialized():
        dist.all_reduce(totals)
    if totals[3] == 0:
        raise ValueError("No samples were available")
    return dict(zip(("loss", "caption", "causal"), (totals[:3]/totals[3]).tolist())), updates


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_gradient_checkpointing(captioner, enabled):
    """Apply the memory setting to both pretrained backbones."""
    language = captioner.decoder.llama
    if enabled:
        language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        language.gradient_checkpointing_disable()
    vision = captioner.encoder.vision_model
    if hasattr(vision, "gradient_checkpointing_enable"):
        if enabled:
            vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            vision.gradient_checkpointing_disable()
    else:
        # Support both the local timm adapter and Transformers timm wrappers.
        backbone = getattr(vision, "timm_model", getattr(vision, "model", vision))
        if hasattr(backbone, "set_grad_checkpointing"):
            backbone.set_grad_checkpointing(enabled)
        elif enabled:
            raise ValueError("This vision encoder does not support gradient checkpointing")


def main(args, stage):
    import yaml
    from .models.factory import build_model
    from .utils.checkpoint import load_checkpoint, save_checkpoint, model_spec
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    if args.encoder_checkpoint:
        config["model"]["encoder"]["model_name"] = args.encoder_checkpoint
    if args.llm_checkpoint:
        config["model"]["decoder"]["model_name"] = args.llm_checkpoint
    if args.encoder_loader:
        config["model"]["encoder"]["loader"] = args.encoder_loader
    if args.data:
        config["data"]["dataset_path"] = args.data
    if args.save_dir:
        config["training"]["save_dir"] = args.save_dir
    rank, world, local = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        seed_all(config["training"]["seed"])
        model = build_model(config, stage, device)
        if stage == 2:
            if not args.stage1_ckpt:
                raise ValueError("Stage 2 requires --stage1-ckpt")
            load_checkpoint(args.stage1_ckpt, model.stage1)
        elif args.init_checkpoint:
            load_checkpoint(args.init_checkpoint, model)
        configure_gradient_checkpointing(model if stage == 1 else model.stage1,
                                         config["training"].get("gradient_checkpointing", True))
        data = config["data"]
        # Batch size is global across ranks, before gradient accumulation.
        if data["batch_size"] % world:
            raise ValueError("Global batch size must be divisible by WORLD_SIZE")
        train = SimplePreprocessedCausalVideoDataset(data["dataset_path"], "train", data["resolution"], num_frames=data["num_frames"])
        val = SimplePreprocessedCausalVideoDataset(data["dataset_path"], "val", data["resolution"], augment=False, num_frames=data["num_frames"])
        sampler = DistributedSampler(train, num_replicas=world, rank=rank, seed=config["training"]["seed"]) if world > 1 else None
        val = Subset(val, range(rank, len(val), world))
        kwargs = dict(batch_size=data["batch_size"]//world, num_workers=data["num_workers"],
                      collate_fn=simple_collate_fn, pin_memory=device.type == "cuda")
        train_loader = DataLoader(train, sampler=sampler, shuffle=sampler is None, **kwargs)
        val_loader = DataLoader(val, shuffle=False, **kwargs)
        phases = STAGE1_PHASES if stage == 1 else STAGE2_PHASES
        epochs = sum(p["epochs"] for p in phases)
        previous, wrapped, best = None, model, float("inf")
        for epoch in range(epochs):
            seed_all(config["training"]["seed"] + epoch * world + rank)
            if sampler:
                sampler.set_epoch(epoch)
            phase = phase_at(epoch, phases)
            if phase != previous:
                if wrapped is not model:
                    del wrapped
                optimizer = build_optimizer(model, phases[phase], stage)
                n_batches = len(train_loader)
                updates = math.ceil(n_batches/config["training"]["gradient_accumulation_steps"])
                scheduler = build_scheduler(optimizer, updates * phases[phase]["epochs"], phases[phase]["warmup"])
                wrapped = DistributedDataParallel(model, device_ids=[local] if device.type == "cuda" else None,
                                                  find_unused_parameters=True) if world > 1 else model
                previous = phase
            training, updates = run_epoch(wrapped, train_loader, optimizer, scheduler, device, config,
                                           stage, True)
            # Use the unwrapped module: validation ranks may have unequal batch counts.
            validation, _ = run_epoch(model, val_loader, None, None, device, config, stage, False)
            if rank == 0:
                log = {"epoch": epoch+1, "phase": phase+1, "updates": updates,
                       "train": training, "validation_objective": validation}
                print(json.dumps(log), flush=True)
                improved = validation["loss"] < best
                best = min(best, validation["loss"])
                state = {"format_version": 2, "stage": stage, "epoch": epoch+1,
                         "model_spec": model_spec(model),
                         "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                         "scheduler_state_dict": scheduler.state_dict(), "config": config,
                         "validation_objective": validation}
                save_checkpoint(state, improved, config["training"]["save_dir"], f"stage{stage}_ep{epoch+1:03d}.pt")
            if world > 1:
                dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
