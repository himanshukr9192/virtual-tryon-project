# Virtual Try-On AI 👗

An AI-powered virtual try-on tool that automatically recognizes clothing items and renders them realistically onto a person's photo using advanced machine learning.

## Features

✨ **Automatic Clothing Classification** - Zero-shot detection using CLIP to identify garment types
🎨 **Semantic Masking** - Intelligent placement of clothing items on the body
🖼️ **High-Quality Rendering** - Uses Stable Diffusion for photorealistic draping
⚡ **Easy-to-Use Interface** - Interactive Gradio UI for quick try-ons
🚀 **Production-Ready** - FastAPI backend with modular architecture

## Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **ML Framework**: PyTorch, HuggingFace Transformers, Diffusers
- **Frontend**: Gradio
- **Image Processing**: OpenCV, Pillow (PIL)
- **ML Models**: 
  - CLIP (Zero-shot classification)
  - Stable Diffusion (Image generation)

## Installation

### Prerequisites

- Python 3.10 or higher
- Git
- NVIDIA GPU (recommended for faster inference)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/virtual-tryon-ai.git
   cd virtual-tryon-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models** (optional, auto-downloaded on first run)
   ```bash
   python backend/models/download_models.py
   ```

## Usage

### Quick Start with Gradio UI

```bash
python frontend/app.py
```

This launches the interactive Gradio interface at `http://localhost:7860`

### Using FastAPI Backend

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Then navigate to `http://localhost:8000/docs` for the API documentation.

## API Endpoints

### POST `/api/virtual-tryon`
Virtual try-on with automatic garment detection.

**Request:**
```json
{
  "person_image": "base64_encoded_image",
  "garment_image": "base64_encoded_image",
  "garment_type": "auto"
}
```

**Response:**
```json
{
  "result_image": "base64_encoded_result",
  "garment_detected": "shirt",
  "confidence": 0.95
}
```

### POST `/api/classify-garment`
Classify clothing type from an image.

**Request:**
```json
{
  "image": "base64_encoded_image"
}
```

**Response:**
```json
{
  "garment_type": "shirt",
  "confidence": 0.87,
  "alternatives": ["t-shirt", "blouse"]
}
```

## Project Structure

```
virtual-tryon-ai/
├── README.md
├── requirements.txt
├── config.py
├── .gitignore
├── LICENSE
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clothing_classifier.py  # CLIP-based classification
│   │   └── download_models.py      # Model download utility
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_processor.py      # Image I/O and preprocessing
│   │   ├── masking.py              # Semantic masking
│   │   └── inference.py            # Model inference pipeline
│   └── routes/
│       ├── __init__.py
│       └── api.py                  # API route handlers
├── frontend/
│   ├── __init__.py
│   └── app.py                  # Gradio interface
└── tests/
    ├── __init__.py
    └── test_models.py         # Unit tests
```

## Configuration

Edit `config.py` to customize:
- Model names and versions
- Image processing parameters
- API settings
- Inference device (CPU/GPU)

## Performance

- **Inference time** (GPU): ~5-10 seconds per image
- **Inference time** (CPU): ~30-60 seconds per image
- **Memory usage**: ~4-6 GB VRAM (GPU)

## Troubleshooting

### Out of Memory Error
- Reduce `MAX_IMAGE_SIZE` in `config.py`
- Use `device = "cpu"` for slower but lower-memory inference
- Enable image optimization in `image_processor.py`

### Slow Inference
- Ensure GPU is being used (check CUDA availability)
- Reduce image resolution
- Use smaller model variants

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this project, please cite:

```bibtex
@software{virtual_tryon_ai_2026,
  title={Virtual Try-On AI},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/virtual-tryon-ai}
}
```

## Acknowledgments

- HuggingFace for TransformersLib and Diffusers
- OpenAI for CLIP model
- StabilityAI for Stable Diffusion
- Gradio team for the excellent UI framework

## Contact

For questions or support, please open an issue on GitHub or contact through the repository.

---

⭐ If you find this project helpful, please consider giving it a star!
