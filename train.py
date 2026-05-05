"""
Entry point for training.

Usage:
    python train.py                          # use default config
    python train.py training.epochs=300     # override any config key
    python train.py --resume checkpoints/ckpt_epoch_0100.pth
"""

import argparse
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import build_dataloader
from src.training.trainer import Trainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Build dataloader
    loader = build_dataloader(
        data_path=cfg.data.dataset_path,
        image_size=cfg.model.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # Train
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", type=str, default=None)
    args, _ = parser.parse_known_args()

    trainer = Trainer(cfg)
    trainer.train(loader, resume=args.resume)


if __name__ == "__main__":
    main()
