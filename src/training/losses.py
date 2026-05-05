"""
GAN loss functions.

Both standard BCE loss and optional WGAN-GP are provided.
For most use-cases, BCE with label smoothing works well.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd


class GANLoss:
    """
    Standard GAN loss with optional label smoothing.
    Uses BCEWithLogitsLoss (numerically stable sigmoid + BCE).
    """

    def __init__(self, label_smoothing: float = 0.1, device: str = "cpu"):
        self.real_label = 1.0 - label_smoothing   # e.g. 0.9
        self.fake_label = 0.0
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

    def _labels(self, size: int, real: bool) -> torch.Tensor:
        value = self.real_label if real else self.fake_label
        return torch.full((size,), value, device=self.device)

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Total discriminator loss = loss_real + loss_fake."""
        loss_real = self.criterion(real_logits, self._labels(real_logits.size(0), real=True))
        loss_fake = self.criterion(fake_logits, self._labels(fake_logits.size(0), real=False))
        total = loss_real + loss_fake
        return total, {"d_real": loss_real.item(), "d_fake": loss_fake.item()}

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator wants discriminator to think fakes are real."""
        return self.criterion(fake_logits, self._labels(fake_logits.size(0), real=True))


# ─────────────────────────────────
# Optional: WGAN-GP (higher quality)
# ─────────────────────────────────

def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: str = "cpu",
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute WGAN-GP gradient penalty.
    Encourages the discriminator to be 1-Lipschitz.
    """
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, 1, device=device).expand_as(real)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    grad = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = grad.view(batch, -1).norm(2, dim=1)
    penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return penalty
