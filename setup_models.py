"""Pre-download InsightFace models and export YOLO ONNX before running the app."""

import os
import sys
from pathlib import Path


def check_and_download_face_models():
    """Download InsightFace models if not present."""
    print("=" * 60)
    print("Checking InsightFace models...")
    print("=" * 60)
    
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("❌ insightface not installed. Run: pip install -r requirements.txt")
        return False
    
    models = ["buffalo_l", "buffalo_s"]
    home = Path.home()
    insightface_root = home / ".insightface" / "models"
    
    for model_name in models:
        model_path = insightface_root / model_name
        if model_path.exists() and any(model_path.glob("*.onnx")):
            print(f"✓ {model_name} already downloaded at {model_path}")
        else:
            print(f"⬇ Downloading {model_name} (~140-250 MB)...")
            try:
                # Trigger download by creating analyzer
                FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
                print(f"✓ {model_name} downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
                return False
    
    print(f"\n✓ All InsightFace models ready at {insightface_root}")
    return True


def check_and_export_yolo_onnx():
    """Export YOLO model to ONNX if not present."""
    print("\n" + "=" * 60)
    print("Checking YOLO ONNX model...")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install -r requirements.txt")
        return False
    
    pt_path = Path("yolov8n.pt")
    onnx_path = Path("yolov8n.onnx")
    
    if onnx_path.exists():
        print(f"✓ yolov8n.onnx already exists ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    
    if not pt_path.exists():
        print(f"⬇ Downloading yolov8n.pt (~6 MB)...")
        try:
            # YOLO will auto-download if model doesn't exist
            model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"❌ Failed to download yolov8n.pt: {e}")
            return False
    
    print(f"🔧 Exporting yolov8n.pt to ONNX format...")
    try:
        model = YOLO("yolov8n.pt")
        model.export(format="onnx", dynamic=True, simplify=False)
        print(f"✓ yolov8n.onnx exported successfully ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ Failed to export ONNX: {e}")
        return False


def main():
    """Check and download all required models."""
    print("\n🚀 StaffLens AI - Model Setup")
    print("This will download required models (~400 MB total)\n")
    
    # Check InsightFace models
    face_ok = check_and_download_face_models()
    
    # Check YOLO ONNX
    yolo_ok = check_and_export_yolo_onnx()
    
    print("\n" + "=" * 60)
    if face_ok and yolo_ok:
        print("✅ All models ready! You can now run:")
        print("   python server.py")
        print("   python app.py")
        print("=" * 60)
        return 0
    else:
        print("❌ Model setup incomplete. Please fix errors above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
