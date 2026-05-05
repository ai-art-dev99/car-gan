"""
GAN Trainer – handles the full training loop.

Usage:
    from src.training.trainer import Trainer
    trainer = Trainer(config)
    trainer.train()
"""

from __future__ import annotations
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.models import Generator, Discriminator
from src.training.losses import GANLoss
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.visualization import save_sample_grid


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = self._get_device()

        # ── Models ──────────────────────────────────────────
        self.G = Generator(
            latent_dim=cfg.model.latent_dim,
            features=cfg.model.generator_features,
            channels=cfg.model.channels,
            image_size=cfg.model.image_size,
        ).to(self.device)

        self.D = Discriminator(
            features=cfg.model.discriminator_features,
            channels=cfg.model.channels,
            image_size=cfg.model.image_size,
        ).to(self.device)

        # ── Optimisers ──────────────────────────────────────
        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=cfg.training.lr_generator,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )
        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=cfg.training.lr_discriminator,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )

        # ── Loss ────────────────────────────────────────────
        self.criterion = GANLoss(
            label_smoothing=cfg.training.label_smoothing,
            device=self.device,
        )

        # Fixed noise for consistent sample grid across epochs
        self.fixed_noise = torch.randn(
            cfg.logging.sample_count,
            cfg.model.latent_dim,
            device=self.device,
        )

        # Directories
        self.log_dir = Path(cfg.logging.log_dir)
        self.ckpt_dir = Path(cfg.logging.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.history: list[dict] = []

    # ────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────

    def train(self, loader: DataLoader, resume: str | None = None) -> None:
        if resume:
            self.start_epoch = load_checkpoint(
                resume, self.G, self.D, self.opt_G, self.opt_D, self.device
            )
            print(f"[Trainer] Resumed from epoch {self.start_epoch}")

        total_epochs = self.cfg.training.epochs
        print(f"[Trainer] Starting training for {total_epochs} epochs on {self.device}")

        for epoch in range(self.start_epoch, total_epochs):
            t0 = time.time()
            stats = self._train_epoch(loader, epoch)
            elapsed = time.time() - t0

            self.history.append(stats)
            print(
                f"Epoch [{epoch+1:>4}/{total_epochs}] "
                f"D: {stats['d_loss']:.4f}  G: {stats['g_loss']:.4f}  "
                f"time: {elapsed:.1f}s"
            )

            # Save sample grid
            if (epoch + 1) % self.cfg.logging.sample_interval == 0:
                save_sample_grid(
                    self.G, self.fixed_noise,
                    path=self.log_dir / f"samples_epoch_{epoch+1:04d}.png",
                    nrow=4,
                )

            # Save checkpoint
            if (epoch + 1) % self.cfg.logging.save_interval == 0:
                save_checkpoint(
                    self.G, self.D, self.opt_G, self.opt_D,
                    epoch=epoch + 1,
                    path=self.ckpt_dir / f"ckpt_epoch_{epoch+1:04d}.pth",
                )

        # Always save final checkpoint
        save_checkpoint(
            self.G, self.D, self.opt_G, self.opt_D,
            epoch=total_epochs,
            path=self.ckpt_dir / "ckpt_final.pth",
        )
        print("[Trainer] Training complete.")

    # ────────────────────────────────────────────────────────
    # Internal
    # ────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.G.train()
        self.D.train()
        total_d, total_g, n_batches = 0.0, 0.0, 0

        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(self.device)
            batch = real_imgs.size(0)

            # ── Train Discriminator ──────────────────────────
            for _ in range(self.cfg.training.n_critic):
                noise = torch.randn(batch, self.cfg.model.latent_dim, device=self.device)
                fake_imgs = self.G(noise).detach()

                real_logits = self.D(real_imgs).squeeze()
                fake_logits = self.D(fake_imgs).squeeze()
                d_loss, _ = self.criterion.discriminator_loss(real_logits, fake_logits)

                self.opt_D.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
                self.opt_D.step()

            # ── Train Generator ──────────────────────────────
            noise = torch.randn(batch, self.cfg.model.latent_dim, device=self.device)
            fake_imgs = self.G(noise)
            fake_logits = self.D(fake_imgs).squeeze()
            g_loss = self.criterion.generator_loss(fake_logits)

            self.opt_G.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
            self.opt_G.step()

            total_d += d_loss.item()
            total_g += g_loss.item()
            n_batches += 1

        return {"d_loss": total_d / n_batches, "g_loss": total_g / n_batches, "epoch": epoch + 1}

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"[Trainer] Using device: {device}")
        return device
