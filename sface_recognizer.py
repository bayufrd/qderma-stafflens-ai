"""Root wrapper for OpenCV Zoo YuNet + SFace (Apache-2.0) face detection & recognition."""

from third_party.opencv_zoo.yunet_sface import (
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
