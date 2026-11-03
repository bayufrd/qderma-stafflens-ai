"""Pre-download Apache-2.0 OpenCV Zoo (YuNet + SFace) and YOLOX ONNX models before running the app."""

import os
import sys


def check_and_download_face_models() -> bool:
    """Download Apache-2.0 YuNet and SFace ONNX models if not present."""
    print("=" * 60)
    print("Checking OpenCV Zoo YuNet + SFace models (Apache-2.0)...")
    print("=" * 60)

    try:
        from sface_recognizer import ensure_opencv_face_models
    except ImportError as e:
        print(f"❌ Failed to import sface_recognizer: {e}")
        return False

    try:
        yunet_path, sface_path = ensure_opencv_face_models()
        print(f"✓ {yunet_path.name} ready ({yunet_path.stat().st_size / 1024:.0f} KB, Apache-2.0)")
        print(f"✓ {sface_path.name} ready ({sface_path.stat().st_size / 1024 / 1024:.1f} MB, Apache-2.0)")
        return True
    except Exception as e:
        print(f"❌ Failed to prepare OpenCV Zoo face models: {e}")
        return False


def check_and_download_yolox_onnx() -> bool:
    """Download Apache-2.0 YOLOX ONNX model if not present."""
    print("\n" + "=" * 60)
    print("Checking YOLOX ONNX model (Apache-2.0)...")
    print("=" * 60)

    try:
        from yolox_detector import ensure_yolox_onnx, resolve_yolox_model_name
    except ImportError as e:
        print(f"❌ Failed to import yolox_detector: {e}")
        return False

    model_name = resolve_yolox_model_name(os.getenv("YOLO_MODEL", "yolox_nano.onnx"))
    try:
        onnx_path = ensure_yolox_onnx(model_name)
        print(f"✓ {onnx_path.name} ready ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB, Apache-2.0)")
        return True
    except Exception as e:
        print(f"❌ Failed to prepare YOLOX ONNX model: {e}")
        return False


def main() -> int:
    """Check and download all required models."""
    print("\n🚀 StaffLens AI - Model Setup (100% Apache-2.0 Commercial-Free Models)")
    print("This will verify/download YuNet, SFace, and YOLOX ONNX models (~41 MB total)\n")

    face_ok = check_and_download_face_models()
    yolo_ok = check_and_download_yolox_onnx()

    print("\n" + "=" * 60)
    if face_ok and yolo_ok:
        print("✅ All Apache-2.0 models ready! You can now run:")
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
