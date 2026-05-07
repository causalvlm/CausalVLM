"""
Stage 2: Multimodal causal head training (two-phase).

Phase 1 (10 ep) — causal head only.   LR: head=1e-4
Phase 2 (10 ep) — full end-to-end.    LR: head=1e-4, proj=1e-5, vis=1e-6, llm=1e-6

Loss: L_total = L_caption + 2 * L_causal
L_causal uses uncapped positive weighting, 20% hard negative mining,
and asymmetric false-negative penalty (lambda_fn=10). See Appendix A.

Usage:
    python train_stage2.py --config configs/stage2.yaml \\
                           --stage1-ckpt checkpoints/stage1/best_model.pt \\
                           --exp-name stage2_v1
"""

import argparse
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageTextToText

from src.data.dataset import SimplePreprocessedCausalVideoDataset, simple_collate_fn
from src.models.perception_lm import PerceptionVideoLM
from src.models.causal_vlm_model import CausalVLM, MultimodalCausalHead, N_MAX_EVENTS
from src.utils.checkpoint import load_checkpoint, save_checkpoint

LAMBDA_CAUSAL  = 2.0
LAMBDA_FN      = 10.0
HNM_RATIO      = 0.20
CAUSAL_THRESH  = 0.5
FROZEN_EPOCHS  = 10
E2E_EPOCHS     = 10
TOTAL_EPOCHS   = FROZEN_EPOCHS + E2E_EPOCHS
GRAD_ACCUM     = 4


def _hnm_mask(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pos_mask = (gt == 1).float()
    neg_mask = gt == 0
    n_neg    = neg_mask.sum().item()
    if n_neg == 0:
        return pos_mask
    masked = pred.clone()
    masked[~neg_mask] = -1e9
    _, idx = torch.topk(masked.flatten(), max(1, int(n_neg * HNM_RATIO)))
    hard   = torch.zeros_like(pred).flatten()
    hard[idx] = 1.0
    return pos_mask + hard.view_as(pred)


def causal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    N    = pred.shape[0]
    mask = torch.triu(torch.ones(N, N, device=pred.device, dtype=torch.bool), diagonal=1)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    p     = pred[mask]
    g     = gt[mask].to(p.dtype)
    n_pos = g.sum()
    n_neg = g.numel() - n_pos
    w_pos = n_neg / n_pos if n_pos > 0 else torch.tensor(1.0, device=p.device)

    M = _hnm_mask(pred.detach(), gt)[mask]

    w           = torch.ones_like(g)
    fn_idx      = (g == 1) & (p < 0.5)
    tp_idx      = (g == 1) & (p >= 0.5)
    w[fn_idx]   = LAMBDA_FN * w_pos
    w[tp_idx]   = w_pos

    bce = F.binary_cross_entropy(p, g, reduction="none")
    return (w * M * bce).sum() / (M.sum() + 1e-7)


def build_optimizer(model: CausalVLM, phase: int) -> torch.optim.Optimizer:
    for p in model.parameters():
        p.requires_grad = False

    for p in model.causal_head.parameters():
        p.requires_grad = True

    if phase == 1:
        return torch.optim.AdamW(
            list(model.causal_head.parameters()),
            lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
        )

    enc  = model.stage1.encoder.vision_model
    proj = model.stage1.decoder.visual_proj
    llm  = model.stage1.decoder.llama
    for m in (enc, proj, llm):
        for p in m.parameters():
            p.requires_grad = True

    return torch.optim.AdamW(
        [
            {"params": list(model.causal_head.parameters()), "lr": 1e-4},
            {"params": list(proj.parameters()),              "lr": 1e-5},
            {"params": list(enc.parameters()),               "lr": 1e-6},
            {"params": list(llm.parameters()),               "lr": 1e-6},
        ],
        betas=(0.9, 0.999), eps=1e-8,
    )


def build_scheduler(optimizer, n_steps: int, warmup: int = 500):
    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def _num_events(batch: dict) -> List[int]:
    if batch.get("num_events"):
        return [int(n) for n in batch["num_events"]]
    masks = batch.get("event_masks")
    if masks is not None:
        return masks.sum(dim=1).tolist()
    adj = batch.get("adjacency_matrix")
    if adj is not None:
        return [adj.shape[1]] * adj.shape[0]
    return []


def train_epoch(model, loader, opt, sched, device, config, epoch):
    model.train()
    tok       = model.stage1.decoder.tokenizer
    total     = cap_t = caus_t = 0.0
    opt.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc=f"Ep {epoch+1} train")):
        frames     = batch["frames"].to(device, dtype=torch.bfloat16)
        num_ev     = _num_events(batch)
        event_desc = batch.get("event_descriptions")
        gt_adj     = batch.get("adjacency_matrix")

        if not num_ev or all(n == 0 for n in num_ev):
            continue

        ids = tok(
            batch["captions"], return_tensors="pt", padding=True,
            truncation=True, max_length=config.data.max_caption_length,
        ).to(device)

        out      = model(frames, event_desc, num_ev, labels=ids.input_ids)
        adj_pred = out["adjacency_matrix"]
        sizes    = out["sizes"]
        cap_s    = out["caption_loss"].mean() if out["caption_loss"] is not None \
                   else torch.tensor(0.0, device=device)

        caus_s = torch.tensor(0.0, device=device)
        if adj_pred is not None and gt_adj is not None:
            losses = []
            for b in range(len(num_ev)):
                N  = int(sizes[b])
                if N < 1:
                    continue
                gt = gt_adj[b, :N, :N].to(device)
                pr = adj_pred[b, :N, :N].float()
                if gt.sum() > 0:
                    losses.append(causal_loss(pr, gt))
            if losses:
                caus_s = torch.stack(losses).mean()

        loss = (cap_s + LAMBDA_CAUSAL * caus_s) / GRAD_ACCUM
        if torch.isfinite(loss):
            loss.backward()
            total  += loss.item() * GRAD_ACCUM
            cap_t  += cap_s.item()
            caus_t += caus_s.item()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.training.clip_grad_norm,
            )
            opt.step()
            sched.step()
            opt.zero_grad()
            if wandb.run:
                wandb.log({"train/loss": loss.item() * GRAD_ACCUM,
                           "train/caption": cap_s.item(),
                           "train/causal": caus_s.item(),
                           "train/lr": sched.get_last_lr()[0], "epoch": epoch})

    n = len(loader)
    return {"loss": total / n, "caption": cap_t / n, "causal": caus_t / n}


def val_epoch(model, loader, device, config, epoch):
    model.eval()
    tok        = model.stage1.decoder.tokenizer
    all_p, all_g, total = [], [], 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Ep {epoch+1} val"):
            frames     = batch["frames"].to(device, dtype=torch.bfloat16)
            num_ev     = _num_events(batch)
            event_desc = batch.get("event_descriptions")
            gt_adj     = batch.get("adjacency_matrix")
            if not num_ev or all(n == 0 for n in num_ev):
                continue

            ids = tok(
                batch["captions"], return_tensors="pt", padding=True,
                truncation=True, max_length=config.data.max_caption_length,
            ).to(device)

            out      = model(frames, event_desc, num_ev, labels=ids.input_ids)
            adj_pred = out["adjacency_matrix"]
            sizes    = out["sizes"]
            cap_s    = out["caption_loss"].mean() if out["caption_loss"] is not None \
                       else torch.tensor(0.0, device=device)

            caus_s = torch.tensor(0.0, device=device)
            if adj_pred is not None and gt_adj is not None:
                losses = []
                for b in range(len(num_ev)):
                    N  = int(sizes[b])
                    if N < 1:
                        continue
                    gt = gt_adj[b, :N, :N].to(device)
                    pr = adj_pred[b, :N, :N].float()
                    mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
                    all_p.append(pr[mask].cpu())
                    all_g.append(gt[mask].cpu())
                    if gt.sum() > 0:
                        losses.append(causal_loss(pr, gt))
                if losses:
                    caus_s = torch.stack(losses).mean()

            total += (cap_s + LAMBDA_CAUSAL * caus_s).item()

    eps = 1e-7
    if all_p:
        pf = torch.cat(all_p)
        gf = torch.cat(all_g).long()
        b  = (pf > CAUSAL_THRESH).long()
        tp = (b & gf).sum().item()
        fp = (b & (1 - gf)).sum().item()
        fn = ((1 - b) & gf).sum().item()
        pr = tp / (tp + fp + eps)
        rc = tp / (tp + fn + eps)
        f1 = 2 * pr * rc / (pr + rc + eps)
    else:
        pr = rc = f1 = 0.0

    m = {"loss": total / max(1, len(loader)), "f1": f1, "precision": pr, "recall": rc}
    if wandb.run:
        wandb.log({f"val/{k}": v for k, v in m.items()} | {"epoch": epoch})
    return m


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

    stage1 = PerceptionVideoLM(config, vision_model=vt).to(torch.bfloat16)
    load_checkpoint(args.stage1_ckpt, stage1)

    text_dim    = stage1.decoder.llama.config.hidden_size
    causal_head = MultimodalCausalHead(
        visual_dim=config.model.encoder.hidden_size,
        text_dim=text_dim,
    ).to(torch.bfloat16)

    model = CausalVLM(stage1, causal_head).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_ds = SimplePreprocessedCausalVideoDataset(config.data.dataset_path, "train", 448)
    val_ds   = SimplePreprocessedCausalVideoDataset(config.data.dataset_path, "val",   448)
    kw = dict(collate_fn=simple_collate_fn, num_workers=config.data.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, config.data.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   config.data.batch_size, shuffle=False, **kw)

    best_f1    = 0.0
    prev_phase = None

    for epoch in range(TOTAL_EPOCHS):
        phase = 1 if epoch < FROZEN_EPOCHS else 2
        if phase != prev_phase:
            actual = model.module if hasattr(model, "module") else model
            opt    = build_optimizer(actual, phase)
            ph_ep  = FROZEN_EPOCHS if phase == 1 else E2E_EPOCHS
            sched  = build_scheduler(opt, ph_ep * len(train_loader))
            prev_phase = phase
            print(f"\n--- Phase {phase} ---")

        tm = train_epoch(model, train_loader, opt, sched, device, config, epoch)
        vm = val_epoch(model, val_loader, device, config, epoch)
        print(f"Ep {epoch+1:3d} | ph={phase} | "
              f"train={tm['loss']:.4f} | "
              f"val_f1={vm['f1']:.4f} | "
              f"prec={vm['precision']:.4f} | rec={vm['recall']:.4f}")

        is_best = vm["f1"] > best_f1
        if is_best:
            best_f1 = vm["f1"]

        actual = model.module if hasattr(model, "module") else model
        save_checkpoint(
            {"epoch": epoch + 1, "phase": phase,
             "model_state_dict": actual.state_dict(),
             "best_val_f1": best_f1, "val_metrics": vm,
             "config": OmegaConf.to_container(config)},
            is_best, config.training.save_dir, f"stage2_ep{epoch+1:03d}.pt",
        )

    wandb.finish()
    print(f"Stage 2 done. Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--exp-name",    default="stage2")
    main(p.parse_args())
