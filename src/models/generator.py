"""
Generator network for Car GAN.
Architecture: DCGAN-style with transposed convolutions.
Input:  latent vector z  (batch, latent_dim)
Output: RGB image        (batch, 3, image_size, image_size)
"""

import torch
import torch.nn as nn


def _block(in_channels: int, out_channels: int, kernel: int = 4,
           stride: int = 2, padding: int = 1) -> nn.Sequential:
    """Upsampling block: ConvTranspose → BatchNorm → ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    """
    DCGAN Generator.

    Latent vector → series of upsampling blocks → final tanh activation.
    Output values are in [-1, 1], normalize your targets the same way.
    """

    def __init__(self, latent_dim: int = 128, features: int = 64, channels: int = 3,
                 image_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # How many upsampling steps to reach image_size from 4x4 base
        # 4 → 8 → 16 → 32 → 64  (for image_size=64, n_up=4)
        import math
        self.n_up = int(math.log2(image_size)) - 2  # e.g. 4 for 64px, 5 for 128px

        # Initial projection: z → (features * 2^n_up) × 4 × 4
        init_features = features * (2 ** self.n_up)
        self.project = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, init_features, 4, 1, 0, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
        )

        # Upsampling layers
        layers = []
        in_f = init_features
        for i in range(self.n_up - 1):
            out_f = in_f // 2
            layers.append(_block(in_f, out_f))
            in_f = out_f

        # Final layer: no BatchNorm, Tanh activation
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_f, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        )

        self.main = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim) → reshape to (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), -1, 1, 1)
        x = self.project(z)
        return self.main(x)

    @torch.no_grad()
    def generate(self, n: int = 1, device: str = "cpu") -> torch.Tensor:
        """Convenience method: sample n images. Returns tensor in [-1, 1]."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self(z)
