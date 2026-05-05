"""
Discriminator network for Car GAN.
Architecture: DCGAN-style with strided convolutions.
Input:  RGB image  (batch, 3, image_size, image_size)
Output: scalar logit per image (batch, 1)
"""

import torch
import torch.nn as nn


def _block(in_channels: int, out_channels: int, kernel: int = 4,
           stride: int = 2, padding: int = 1, batch_norm: bool = True) -> nn.Sequential:
    """Downsampling block: Conv → [BatchNorm] → LeakyReLU."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=not batch_norm),
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator.

    Accepts an image and outputs a raw logit (no sigmoid).
    Use BCEWithLogitsLoss during training.
    """

    def __init__(self, features: int = 64, channels: int = 3, image_size: int = 64):
        super().__init__()
        import math
        n_down = int(math.log2(image_size)) - 2  # same number as generator upsamples

        # First layer: no BatchNorm (input is raw image)
        layers: list[nn.Module] = [_block(channels, features, batch_norm=False)]
        in_f = features
        for _ in range(n_down - 1):
            out_f = in_f * 2
            layers.append(_block(in_f, out_f))
            in_f = out_f

        # Final layer: 4×4 → 1×1 scalar
        layers.append(
            nn.Conv2d(in_f, 1, 4, 1, 0, bias=False)
        )

        self.main = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        return out.view(out.size(0), -1)  # (batch, 1) → (batch,)
