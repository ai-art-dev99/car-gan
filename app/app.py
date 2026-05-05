"""
Gradio app – Car GAN image generator.
Deployed on HuggingFace Spaces (SDK: gradio).

Directory structure expected on the Space:
    app.py
    src/
    generator_weights.pth   ← uploaded separately or loaded from HF Hub
"""

import os
import io
import random
import torch
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image

# Add project root to path so `src` is importable
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.models.generator import Generator

# ─────────────────────────────────────────────
# Config (keep in sync with your config.yaml)
# ─────────────────────────────────────────────
LATENT_DIM   = 128
IMAGE_SIZE   = 64
FEATURES     = 64
CHANNELS     = 3
MODEL_REPO   = os.getenv("MODEL_REPO", "your-hf-username/car-gan")
WEIGHT_FILE  = "generator_weights.pth"

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(
    latent_dim=LATENT_DIM,
    features=FEATURES,
    channels=CHANNELS,
    image_size=IMAGE_SIZE,
).to(device)


def _load_weights():
    """Load generator weights from local file or HF Hub."""
    local_path = WEIGHT_FILE
    if not os.path.exists(local_path):
        print(f"[App] Downloading weights from {MODEL_REPO} …")
        local_path = hf_hub_download(repo_id=MODEL_REPO, filename=WEIGHT_FILE)

    ckpt = torch.load(local_path, map_location=device)
    # Support both raw state_dict and full checkpoint
    state = ckpt.get("generator_state", ckpt)
    generator.load_state_dict(state)
    generator.eval()
    print("[App] Generator ready.")


_load_weights()


# ─────────────────────────────────────────────
# Generation logic
# ─────────────────────────────────────────────

def generate_cars(n_images: int, seed: int) -> list[Image.Image]:
    """Generate n_images car images. Returns list of PIL Images."""
    if seed == -1:
        seed = random.randint(0, 2**31)
    torch.manual_seed(seed)
    np.random.seed(seed)

    with torch.no_grad():
        z = torch.randn(int(n_images), LATENT_DIM, device=device)
        imgs = generator(z).cpu()

    # De-normalise [-1, 1] → [0, 255]
    imgs = ((imgs + 1) / 2 * 255).clamp(0, 255).byte()
    pil_images = [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in imgs]
    return pil_images


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

CSS = """
#generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 1.1rem;
    padding: 0.75rem 2rem;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: opacity 0.2s;
    font-weight: 600;
}
#generate-btn:hover { opacity: 0.88; }
.gallery-item img { border-radius: 8px; }
"""

with gr.Blocks(css=CSS, title="Car GAN Generator") as demo:
    gr.Markdown(
        """
        # 🚗 Car GAN — AI Image Generator
        Generate realistic car images using a Deep Convolutional GAN trained on the Stanford Cars dataset.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            n_slider = gr.Slider(
                minimum=1, maximum=16, step=1, value=4,
                label="Number of images",
            )
            seed_input = gr.Number(
                value=-1, label="Seed (-1 = random)", precision=0,
            )
            gen_btn = gr.Button("✨ Generate Cars", elem_id="generate-btn")

            gr.Markdown(
                """
                **Tips**
                - Use a fixed seed to reproduce the same images
                - Generate up to 16 images at once
                """
            )

        with gr.Column(scale=3):
            gallery = gr.Gallery(
                label="Generated cars",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
            )

    gen_btn.click(
        fn=generate_cars,
        inputs=[n_slider, seed_input],
        outputs=gallery,
    )

    gr.Examples(
        examples=[[4, 42], [9, 123], [16, 999]],
        inputs=[n_slider, seed_input],
        fn=generate_cars,
        outputs=gallery,
        cache_examples=True,
    )

    gr.Markdown(
        """
        ---
        Model architecture: DCGAN | Dataset: Stanford Cars | Framework: PyTorch
        """
    )

if __name__ == "__main__":
    demo.launch()
