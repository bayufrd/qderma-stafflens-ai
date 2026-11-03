"""OpenCV Zoo YuNet + SFace (Apache-2.0) face detection and recognition package."""

from .yunet_sface import (
    DEFAULT_FACE_MODEL,
    SFACE_ONNX_FILENAME,
    YUNET_ONNX_FILENAME,
    DetectedFace,
    FaceAnalysis,
    SFaceAnalyzer,
    ensure_opencv_face_models,
    resolve_face_model_name,
)

__all__ = [
    "DEFAULT_FACE_MODEL",
    "SFACE_ONNX_FILENAME",
    "YUNET_ONNX_FILENAME",
    "DetectedFace",
    "FaceAnalysis",
    "SFaceAnalyzer",
    "ensure_opencv_face_models",
    "resolve_face_model_name",
]
