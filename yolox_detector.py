"""Commercial-free (Apache-2.0) YOLOX ONNX Runtime person detector wrapper for StaffLens AI."""

from third_party.yolox import (
    YOLOX_RELEASE_URLS,
    YOLOXDetector,
    demo_postprocess,
    ensure_yolox_onnx,
    multiclass_nms,
    nms,
    preproc,
    resolve_yolox_model_name,
)

__all__ = [
    "YOLOX_RELEASE_URLS",
    "YOLOXDetector",
    "demo_postprocess",
    "ensure_yolox_onnx",
    "multiclass_nms",
    "nms",
    "preproc",
    "resolve_yolox_model_name",
]
