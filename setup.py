"""Setup configuration for Virtual Try-On AI."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="virtual-tryon-ai",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered virtual clothing try-on with automatic garment detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/virtual-tryon-ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "diffusers>=0.20.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "gradio>=4.15.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "virtual-tryon-gradio=frontend.app:main",
            "virtual-tryon-api=backend.main:main",
        ],
    },
)
