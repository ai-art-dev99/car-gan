"""Visualization helpers – save sample grids and plot loss curves."""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import numpy as np


def save_sample_grid(
    generator: nn.Module,
    noise: torch.Tensor,
    path: str | Path,
    nrow: int = 4,
) -> None:
    """Generate images from fixed noise and save as a grid PNG."""
    generator.eval()
    with torch.no_grad():
        fake = generator(noise).cpu()
    # De-normalise from [-1, 1] → [0, 1]
    fake = (fake + 1) / 2
    save_image(fake, str(path), nrow=nrow, padding=2)
    generator.train()
    print(f"[Viz] Sample grid saved → {path}")


def tensor_to_pil(tensor: torch.Tensor):
    """Convert a single image tensor [-1,1] to a PIL Image."""
    from PIL import Image
    img = (tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def plot_loss_curves(history: list[dict], save_path: str | Path | None = None):
    """Plot generator and discriminator loss curves using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Viz] matplotlib not installed – skipping loss plot.")
        return

    epochs = [h["epoch"] for h in history]
    g_loss = [h["g_loss"] for h in history]
    d_loss = [h["d_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, g_loss, label="Generator", linewidth=1.5)
    ax.plot(epochs, d_loss, label="Discriminator", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Losses")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
        print(f"[Viz] Loss curve saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
