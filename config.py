"""Configuration file for Virtual Try-On AI application."""

import os
from typing import Literal

# Model Configuration
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-inpainting"
DEVICE = "cuda" if os.environ.get("USE_GPU", "true").lower() == "true" else "cpu"

# Image Processing
MAX_IMAGE_SIZE = (768, 1024)  # (width, height)
MIN_IMAGE_SIZE = (256, 256)
IMAGE_QUALITY = 95

# Inference Configuration
INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
SEED = 42

# Clothing Classification
GARMENT_TYPES = [
    "shirt",
    "t-shirt",
    "blouse",
    "dress",
    "pants",
    "jeans",
    "skirt",
    "jacket",
    "coat",
    "sweater",
    "hoodie",
    "hat",
    "cap",
    "shoes",
    "glasses",
    "sunglasses",
    "scarf",
    "tie",
]

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True
API_LOG_LEVEL = "info"

# Gradio Configuration
GRADIO_HOST = "0.0.0.0"
GRADIO_PORT = 7860
GRADIO_SHARE = False
GRADIO_THEME = "default"

# Model Cache
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "virtual_tryon_ai")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Semantic Masking
MASK_DILATION_KERNEL = 15
MASK_BLUR_SIZE = (5, 5)
CONFIDENCE_THRESHOLD = 0.5

# Logging
LOG_FILE = "logs/app.log"
DEBUG_MODE = False

# Application Info
APP_NAME = "Virtual Try-On AI"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-powered virtual clothing try-on with automatic garment detection"
