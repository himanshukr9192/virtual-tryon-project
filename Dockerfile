# Use NVIDIA CUDA image for GPU support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Expose ports
EXPOSE 7860 8000

# Default to Gradio interface
CMD ["python", "frontend/app.py"]

# For FastAPI: CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
