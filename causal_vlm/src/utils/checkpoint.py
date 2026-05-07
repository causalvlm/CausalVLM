import torch
from pathlib import Path


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, filename: str) -> None:
    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / filename)
    if is_best:
        torch.save(state, out / "best_model.pt")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    load_optimizer_state: bool = True,
) -> dict:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    if load_optimizer_state:
        if optimizer and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"  Could not load optimizer state: {e}")
        if scheduler and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"  Could not load scheduler state: {e}")

    return ckpt
