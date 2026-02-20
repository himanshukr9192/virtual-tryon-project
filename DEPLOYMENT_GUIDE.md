# Deployment Guide

Complete guide for deploying Virtual Try-On AI to GitHub and other platforms.

## Table of Contents

1. [GitHub Repository Setup](#github-repository-setup)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [GitHub Actions CI/CD](#github-actions-cicd)

## GitHub Repository Setup

### Prerequisites

- GitHub account
- Git installed locally
- Python 3.10+

### Steps

1. **Create a new GitHub repository**
   ```bash
   # Go to https://github.com/new and create a repository
   # Name it: virtual-tryon-ai
   ```

2. **Initialize local repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Virtual Try-On AI project"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/virtual-tryon-ai.git
   git push -u origin main
   ```

3. **Add collaborators (optional)**
   - Go to repository Settings → Collaborators
   - Add team members

## Local Development

### First-Time Setup

```bash
# Run the quick start script
python quickstart.py

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Development Workflow

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Make changes to code

# Format code with black
black backend frontend

# Lint code
flake8 backend frontend

# Run type checking
mypy backend frontend

# Run tests
pytest tests/ -v

# Commit and push
git add .
git commit -m "Describe your changes"
git push origin main
```

## Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t virtual-tryon-ai:latest .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.10 -t virtual-tryon-ai:1.0.0 .
```

### Run with Docker

```bash
# Run Gradio interface
docker run --gpus all -p 7860:7860 virtual-tryon-ai:latest

# Run FastAPI server
docker run --gpus all -p 8000:8000 \
  virtual-tryon-ai:latest \
  uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
# Start both services
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag image
docker tag virtual-tryon-ai:latest yourname/virtual-tryon-ai:latest

# Push image
docker push yourname/virtual-tryon-ai:latest
```

## Cloud Deployment

### AWS EC2

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: g4dn.xlarge (GPU) or larger
   - Security Group: Allow ports 22, 7860, 8000

2. **Connect via SSH**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install NVIDIA drivers**
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-525
   sudo apt-get install -y nvidia-docker2
   ```

4. **Clone repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/virtual-tryon-ai.git
   cd virtual-tryon-ai
   python quickstart.py
   ```

5. **Run application**
   ```bash
   source venv/bin/activate
   python frontend/app.py --share
   ```

### Google Colab

```python
# Install dependencies
!pip install -r requirements.txt

# Download models
!python backend/models/download_models.py

# Run Gradio with public link
!python frontend/app.py --share
```

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Choose Gradio as the SDK
3. Push your repository:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/virtual-tryon-ai
   cd virtual-tryon-ai
   cp ../virtual-tryon-ai/* .
   git add .
   git commit -m "Initial upload"
   git push
   ```

### Heroku (Note: GPU not available free tier)

```bash
# Install Heroku CLI and login
heroku login

# Create Procfile
cat > Procfile << EOF
web: python -m uvicorn backend.main:app --host 0.0.0.0 --port \$PORT
EOF

# Deploy
heroku create virtual-tryon-ai
git push heroku main
```

## GitHub Actions CI/CD

The project includes automated testing and building via GitHub Actions.

### Current Workflows

**Tests and Linting** (`.github/workflows/tests.yml`)
- Runs on every push and pull request
- Tests Python 3.10 and 3.11
- Performs linting and type checking
- Builds distribution package on main branch

### View Workflow Status

1. Go to your GitHub repository
2. Click "Actions" tab
3. View workflow runs and logs

### Custom Workflows

Add new workflows to `.github/workflows/`:

```yaml
name: Custom Workflow

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  custom-job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run custom command
        run: echo "Hello from GitHub Actions!"
```

## Performance Optimization

### Model Optimization

```python
# Use quantization for faster inference
from diffusers import StableDiffusionInpaintPipeline
import torch

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,  # Use half precision
)
```

### Caching for CI/CD

```yaml
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

## Monitoring and Logging

### Application Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check with authentication (if added)
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/health
```

## Backup and Recovery

```bash
# Create backup
tar -czf virtual-tryon-ai-backup-$(date +%Y%m%d).tar.gz \
  --exclude='venv' \
  --exclude='.git' \
  --exclude='__pycache__' \
  .

# Create git backup
git push origin main
git push origin --all
```

## Troubleshooting

### Out of GPU Memory
```python
# Reduce model precision
torch_dtype=torch.float32  # Instead of float16

# Use CPU
device = "cpu"
```

### Slow Inference
```python
# Enable attention slicing
pipeline.enable_attention_slicing()

# Or xFormers optimization
pipeline.enable_xformers_memory_efficient_attention()
```

## Support

For issues and questions:
- Open GitHub Issues
- Check documentation in README.md
- Review code comments

---

**Last Updated**: February 2026
