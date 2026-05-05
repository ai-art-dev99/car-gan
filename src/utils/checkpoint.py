"""Checkpoint save/load helpers."""

from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: Optimizer,
    opt_d: Optimizer,
    epoch: int,
    path: str | Path,
) -> None:
    path = Path(path)
    torch.save(
        {
            "epoch": epoch,
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "opt_g_state": opt_g.state_dict(),
            "opt_d_state": opt_d.state_dict(),
        },
        path,
    )
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: Optimizer,
    opt_d: Optimizer,
    device: torch.device | str = "cpu",
) -> int:
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator_state"])
    discriminator.load_state_dict(ckpt["discriminator_state"])
    opt_g.load_state_dict(ckpt["opt_g_state"])
    opt_d.load_state_dict(ckpt["opt_d_state"])
    epoch = ckpt.get("epoch", 0)
    print(f"[Checkpoint] Loaded ← {path}  (epoch {epoch})")
    return epoch


def load_generator_only(
    path: str | Path,
    generator: nn.Module,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load only the generator weights – used at inference time."""
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator_state"])
    generator.eval()
    print(f"[Checkpoint] Generator loaded ← {path}")
    return generator
