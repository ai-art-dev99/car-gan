# Car GAN — Synthetic Car Image Generator 🚗

A Deep Convolutional GAN (DCGAN) trained from scratch on the Stanford Cars dataset to generate novel, photorealistic car images from random noise.

🔗 **Live Demo:** [HuggingFace Spaces](https://parsa2025ai-cargandemo.hf.space)

---

## What it does

Press a button — the generator network samples from a random noise vector and produces a car image that has never existed. Every generation is unique.

This project explores the fundamentals of generative modelling: how a generator and discriminator train against each other until the generator learns the distribution of real images well enough to fool the discriminator.

---

## Model Details

| Property | Value |
|---|---|
| Architecture | Deep Convolutional GAN (DCGAN) |
| Dataset | Stanford Cars (~16,000 images) |
| Output resolution | 64×64 RGB |
| Framework | PyTorch |
| Demo | Gradio on HuggingFace Spaces |

---

## Project Structure

```
car-gan/
├── src/              # Generator and Discriminator model definitions
├── configs/          # Training hyperparameters
├── notebooks/
│   └── train_colab.ipynb   # Full training pipeline (Google Colab)
├── app/
│   └── app.py        # Gradio demo
└── train.py          # Training entry point
```

---

## Getting Started

```bash
git clone https://github.com/ai-art-dev99/car-gan.git
cd car-gan

pip install -r requirements.txt

# Run the demo locally
python app/app.py
```

**To retrain the model:**
```bash
pip install -r requirements_train.txt
python train.py
```

Or open `notebooks/train_colab.ipynb` in Google Colab for a GPU-accelerated training pipeline.

---

## How DCGAN Works

```
Random noise vector (z)
        ↓
Generator (series of transposed convolutions)
        ↓
Fake image (64×64)
        ↓
Discriminator tries to distinguish real vs fake
        ↑
Real images from Stanford Cars dataset
```

Generator and discriminator train adversarially until the generator produces images indistinguishable from real ones.

---

## Author

**Amirparsa Rouhi** · [aprouhi.com](https://aprouhi.com) · [LinkedIn](https://linkedin.com/in/amirparsa-rouhi)
