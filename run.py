#!/usr/bin/env python3
"""
StaffLens AI - One-command launcher
Checks environment, installs dependencies, downloads models, then runs server.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str, check: bool = True) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        sys.exit(result.returncode)
    return result.returncode


def check_python_version():
    """Ensure Python 3.9+."""
    if sys.version_info < (3, 9):
        print(f"❌ Python 3.9+ required. Current: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def check_or_create_venv():
    """Check if venv exists, create if not."""
    venv_dir = Path(".venv")
    venv_python = venv_dir / "bin" / "python" if os.name != "nt" else venv_dir / "Scripts" / "python.exe"
    
    if venv_python.exists():
        print(f"✓ Virtual environment found at {venv_dir}")
        return str(venv_python)
    
    print(f"⚠️  Virtual environment not found. Creating at {venv_dir}...")
    run_command([sys.executable, "-m", "venv", str(venv_dir)], "Creating virtual environment")
    return str(venv_python)


def install_requirements(venv_python: str):
    """Install dependencies from requirements.txt."""
    requirements = Path("requirements.txt")
    if not requirements.exists():
        print("❌ requirements.txt not found")
        sys.exit(1)
    
    # Check if already installed (simple marker check)
    marker = Path(".venv") / ".requirements_installed"
    if marker.exists():
        print("✓ Requirements already installed (to reinstall: rm .venv/.requirements_installed)")
        return
    
    run_command(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip"
    )
    
    run_command(
        [venv_python, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing requirements (~200 MB, may take 2-5 minutes)"
    )
    
    # Create marker
    marker.write_text(f"installed at {subprocess.check_output(['date']).decode().strip()}\n")
    print("✓ Requirements installed successfully")


def download_models(venv_python: str):
    """Download AI models if not present."""
    setup_models = Path("setup_models.py")
    if not setup_models.exists():
        print("⚠️  setup_models.py not found, skipping model pre-download")
        print("   Models will be downloaded automatically on first use")
        return
    
    # Check if models already exist
    home = Path.home()
    buffalo_s = home / ".insightface" / "models" / "buffalo_s"
    yolo_onnx = Path("yolov8n.onnx")
    
    if buffalo_s.exists() and yolo_onnx.exists():
        print("✓ AI models already downloaded")
        return
    
    print("⬇ Downloading AI models (~400 MB, may take 5-10 minutes)...")
    exit_code = run_command(
        [venv_python, "setup_models.py"],
        "Downloading InsightFace and YOLO models",
        check=False
    )
    
    if exit_code != 0:
        print("⚠️  Model download failed, but continuing (models will auto-download on first use)")


def check_env_file():
    """Check if .env exists."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print(f"✓ .env file found")
        return
    
    if env_example.exists():
        print("⚠️  .env not found. Copying from .env.example...")
        env_file.write_text(env_example.read_text())
        print(f"✓ Created .env from .env.example")
        print("   ⚠️  IMPORTANT: Edit .env and set DATABASE_URL and DASHBOARD_PASSWORD")
    else:
        print("❌ Neither .env nor .env.example found")
        sys.exit(1)


def run_server(venv_python: str):
    """Start the FastAPI server."""
    print("\n" + "="*60)
    print("🚀 Starting StaffLens AI server...")
    print("="*60)
    print("\nServer will start at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop\n")
    
    # Use exec to replace process (logs will stream directly)
    os.execv(venv_python, [
        venv_python,
        "-m", "uvicorn",
        "server:app",
        "--host", "127.0.0.1",
        "--port", os.getenv("DASHBOARD_PORT", "8000"),
        "--log-level", "info"
    ])


def main():
    """Main launcher sequence."""
    print("\n🚀 StaffLens AI - One-Command Launcher\n")
    
    # 1. Check Python version
    check_python_version()
    
    # 2. Check/create venv
    venv_python = check_or_create_venv()
    
    # 3. Install requirements
    install_requirements(venv_python)
    
    # 4. Download models
    download_models(venv_python)
    
    # 5. Check .env
    check_env_file()
    
    # 6. Run server (replaces this process)
    run_server(venv_python)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
