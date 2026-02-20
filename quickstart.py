"""Quick start script for Virtual Try-On AI."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: list, description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        command: Command to run as list
        description: Description of the command

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"📌 {description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(command, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        return False


def main():
    """Run the quick start sequence."""
    print("""
    ╔══════════════════════════════════════════════════────────╗
    ║        🎨 Virtual Try-On AI - Quick Start Guide 🎨       ║
    ╚══════════════════════════════════════════════════────────╝
    """)

    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required!")
        sys.exit(1)

    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        if run_command([sys.executable, "-m", "venv", "venv"], "Creating virtual environment"):
            print("✅ Virtual environment created")
        else:
            print("❌ Failed to create virtual environment")
            sys.exit(1)

    # Activate virtual environment
    if sys.platform == "win32":
        activate_cmd = str(venv_path / "Scripts" / "activate.bat")
    else:
        activate_cmd = str(venv_path / "bin" / "activate")

    print(f"\n💡 Activate environment with: source {activate_cmd}")

    # Install requirements
    pip_cmd = str(venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pip")
    if run_command([pip_cmd, "install", "-r", "requirements.txt"], "Installing dependencies"):
        print("✅ Dependencies installed")
    else:
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Download models
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  📥 Downloading Pre-trained Models (Optional)            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    response = input(
        "Do you want to download pre-trained models now? (y/n, ~6GB required): "
    ).lower()

    if response == "y":
        python_cmd = str(venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "python")
        if run_command(
            [python_cmd, "backend/models/download_models.py"],
            "Downloading ML models",
        ):
            print("✅ Models downloaded successfully")
        else:
            print("⚠️  Some models might be downloaded on first run")

    # Display next steps
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  ✅ Setup Complete! Next Steps:                          ║
    ╚══════════════════════════════════════════════════════════╝

    1️⃣  Activate the virtual environment:
        Windows: venv\\Scripts\\activate
        Linux/Mac: source venv/bin/activate

    2️⃣  Start the Gradio interface (recommended for UI):
        python frontend/app.py
        Then open: http://localhost:7860

    3️⃣  OR Start the FastAPI server:
        python -m uvicorn backend.main:app --reload
        Then open: http://localhost:8000/docs

    4️⃣  Run tests (optional):
        pytest tests/ -v

    📚 For more information, see README.md
    """)


if __name__ == "__main__":
    main()
