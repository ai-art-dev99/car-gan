---
title: Car GAN Generator
emoji: 🚗
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.31.0"
app_file: app/app.py
pinned: false
license: mit
---

# Car GAN – AI Car Image Generator

Generate synthetic car images with a DCGAN trained on the Stanford Cars dataset.

## How it works
1. Press **Generate Cars**
2. The generator network produces novel car images from random noise
3. Download any image you like

## Model details
- Architecture: Deep Convolutional GAN (DCGAN)
- Dataset: Stanford Cars (~16k images)
- Output: 64×64 RGB images
- Framework: PyTorch

## Training
See `notebooks/train_colab.ipynb` for the full training pipeline.
