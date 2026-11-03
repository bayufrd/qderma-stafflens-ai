"""Apache-2.0 Face Detection (YuNet) & Face Recognition (SFace) from OpenCV Zoo.

Both source code and pretrained ONNX weights (`face_detection_yunet_2023mar.onnx` and
`face_recognition_sface_2021dec.onnx`) are licensed under Apache-2.0 and are 100%
free for commercial and SaaS distribution.
"""

from __future__ import annotations

import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

DEFAULT_FACE_MODEL = "sface"
YUNET_ONNX_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_ONNX_FILENAME = "face_recognition_sface_2021dec.onnx"

YUNET_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

_LEGACY_FACE_MODEL_MAP = {
    "buffalo_l": "sface",
    "buffalo_s": "sface",
    "buffalo_m": "sface",
    "buffalo_sc": "sface",
    "antelopev2": "sface",
    "yunet_sface": "sface",
    "opencv_sface": "sface",
    "sface": "sface",
}

_DOWNLOAD_LOCK = threading.Lock()


def resolve_face_model_name(raw_name: str | None) -> str:
    """Normalize configured face model name (mapping legacy InsightFace names to 'sface')."""
    cleaned = (raw_name or DEFAULT_FACE_MODEL).strip().lower()
    if not cleaned:
        return DEFAULT_FACE_MODEL
    return _LEGACY_FACE_MODEL_MAP.get(cleaned, cleaned)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_opencv_face_models(base_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Ensure Apache-2.0 YuNet and SFace ONNX files exist locally, downloading if missing."""
    root = Path(base_dir) if base_dir is not None else _repo_root()
    root.mkdir(parents=True, exist_ok=True)
    yunet_path = root / YUNET_ONNX_FILENAME
    sface_path = root / SFACE_ONNX_FILENAME

    with _DOWNLOAD_LOCK:
        for path, url, min_bytes in (
            (yunet_path, YUNET_DOWNLOAD_URL, 100_000),
            (sface_path, SFACE_DOWNLOAD_URL, 10_000_000),
        ):
            if not path.exists() or path.stat().st_size < min_bytes:
                tmp_path = path.with_suffix(".onnx.part")
                urllib.request.urlretrieve(url, tmp_path)
                tmp_path.replace(path)

    return yunet_path, sface_path


@dataclass
class DetectedFace:
    """Face detection + recognition result compatible with StaffLens pipeline."""

    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    embedding: np.ndarray
    normed_embedding: np.ndarray


class SFaceAnalyzer:
    """Drop-in Apache-2.0 replacement for FaceAnalysis using OpenCV Zoo YuNet + SFace."""

    def __init__(
        self,
        name: str = DEFAULT_FACE_MODEL,
        providers: Sequence[str] | None = None,
        allowed_modules: Sequence[str] | None = None,
        score_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ) -> None:
        self.name = resolve_face_model_name(name)
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.allowed_modules = list(allowed_modules) if allowed_modules else ["detection", "recognition"]
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.det_size: tuple[int, int] = (320, 320)
        self._lock = threading.Lock()

        self.yunet_path, self.sface_path = ensure_opencv_face_models()
        self._detector = cv2.FaceDetectorYN.create(
            str(self.yunet_path),
            "",
            self.det_size,
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(self.sface_path), "")
        self.models = {
            "detection": str(self.yunet_path.name),
            "recognition": str(self.sface_path.name),
        }

    def prepare(self, ctx_id: int = 0, det_size: tuple[int, int] = (320, 320), det_thresh: float | None = None) -> None:
        del ctx_id
        self.det_size = (max(64, int(det_size[0])), max(64, int(det_size[1])))
        if det_thresh is not None:
            self.score_threshold = float(det_thresh)
            self._detector.setScoreThreshold(self.score_threshold)

    def get(self, img: np.ndarray, max_num: int = 0) -> list[DetectedFace]:
        """Detect faces in BGR image and extract 128-D L2-normalized SFace embeddings."""
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        if h < 16 or w < 16:
            return []

        with self._lock:
            # Resize overly large images for detection speed while preserving aspect ratio
            max_side = max(self.det_size[0], self.det_size[1], 640)
            scale = 1.0
            det_img = img
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                new_w = max(16, int(round(w * scale)))
                new_h = max(16, int(round(h * scale)))
                det_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                self._detector.setInputSize((new_w, new_h))
            else:
                self._detector.setInputSize((w, h))

            _, raw_faces = self._detector.detect(det_img)
            if raw_faces is None or len(raw_faces) == 0:
                return []

            faces_arr = np.asarray(raw_faces, dtype=np.float32)
            if scale != 1.0:
                faces_arr[:, :14] /= scale

            # Sort by largest bounding box area * confidence first
            areas = faces_arr[:, 2] * faces_arr[:, 3] * faces_arr[:, 14]
            order = np.argsort(areas)[::-1]
            if max_num > 0:
                order = order[:max_num]

            results: list[DetectedFace] = []
            for idx in order:
                row = faces_arr[idx]
                x, y, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                bbox = np.array([x, y, x + bw, y + bh], dtype=np.float32)
                kps = row[4:14].reshape(5, 2).astype(np.float32)
                det_score = float(row[14])

                aligned = self._recognizer.alignCrop(img, row)
                feat = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
                norm = float(np.linalg.norm(feat)) + 1e-12
                normed = feat / norm

                results.append(
                    DetectedFace(
                        bbox=bbox,
                        kps=kps,
                        det_score=det_score,
                        embedding=normed,
                        normed_embedding=normed,
                    )
                )
            return results


# Alias for drop-in compatibility
FaceAnalysis = SFaceAnalyzer
