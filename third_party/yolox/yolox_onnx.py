"""YOLOX ONNX Runtime inference engine.

Adapted from Megvii-BaseDetection/YOLOX (https://github.com/Megvii-BaseDetection/YOLOX)
Copyright (c) Megvii, Inc. and its affiliates.
Licensed under the Apache License, Version 2.0 (commercial-friendly, permissive open source).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
import requests
import supervision as sv

YOLOX_RELEASE_BASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
YOLOX_RELEASE_URLS: dict[str, str] = {
    "yolox_nano.onnx": f"{YOLOX_RELEASE_BASE}/yolox_nano.onnx",
    "yolox_tiny.onnx": f"{YOLOX_RELEASE_BASE}/yolox_tiny.onnx",
    "yolox_s.onnx": f"{YOLOX_RELEASE_BASE}/yolox_s.onnx",
    "yolox_m.onnx": f"{YOLOX_RELEASE_BASE}/yolox_m.onnx",
    "yolox_l.onnx": f"{YOLOX_RELEASE_BASE}/yolox_l.onnx",
}

# Map legacy YOLOv8 model filenames (AGPL-3.0) or shorthand aliases to Apache-2.0 YOLOX ONNX models
_LEGACY_MODEL_MAP: dict[str, str] = {
    "yolov8n.pt": "yolox_nano.onnx",
    "yolov8n.onnx": "yolox_nano.onnx",
    "yolov8s.pt": "yolox_s.onnx",
    "yolov8s.onnx": "yolox_s.onnx",
    "yolox_nano": "yolox_nano.onnx",
    "yolox-nano": "yolox_nano.onnx",
    "yolox_tiny": "yolox_tiny.onnx",
    "yolox-tiny": "yolox_tiny.onnx",
    "yolox_s": "yolox_s.onnx",
    "yolox-s": "yolox_s.onnx",
}


def resolve_yolox_model_name(raw_name: str | None) -> str:
    """Resolve user/env model setting to an Apache-2.0 YOLOX ONNX filename or path."""
    cleaned = (raw_name or "").strip()
    if not cleaned:
        return "yolox_nano.onnx"
    base = Path(cleaned).name.lower()
    if base in _LEGACY_MODEL_MAP:
        mapped = _LEGACY_MODEL_MAP[base]
        if base.startswith("yolov8"):
            print(f"[YOLOX] Mengalihkan konfigurasi lama '{cleaned}' (AGPL) -> '{mapped}' (Apache-2.0).")
        return mapped
    if cleaned.lower().endswith(".pt"):
        return "yolox_nano.onnx"
    return cleaned


def ensure_yolox_onnx(model_name_or_path: str | Path = "yolox_nano.onnx") -> Path:
    """Ensure the requested YOLOX ONNX model exists on disk, downloading from official Apache-2.0 releases if needed."""
    resolved = resolve_yolox_model_name(str(model_name_or_path))
    path = Path(resolved)
    if not path.is_absolute() and not path.exists():
        root_candidate = Path(__file__).resolve().parents[2] / path
        if root_candidate.exists():
            path = root_candidate

    if not path.exists():
        filename = path.name
        url = YOLOX_RELEASE_URLS.get(filename)
        if not url:
            raise FileNotFoundError(
                f"Model YOLOX '{path}' tidak ditemukan. Pilihan model Apache-2.0: {', '.join(YOLOX_RELEASE_URLS)}"
            )
        target = Path(__file__).resolve().parents[2] / filename if not path.is_absolute() else path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_suffix(".onnx.part")
        print(f"[YOLOX] Mengunduh model Apache-2.0 {filename} dari {url} ...")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(tmp_target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        tmp_target.replace(target)
        path = target

    return path


def preproc(img: np.ndarray, input_size: tuple[int, int], swap: tuple[int, int, int] = (2, 0, 1)) -> tuple[np.ndarray, float]:
    """Letterbox-pad BGR image to `input_size` (H, W) with value 114 and transpose to CHW float32."""
    if len(img.shape) == 3:
        padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    else:
        padded_img = np.full(input_size, 114, dtype=np.uint8)

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    new_w = max(1, int(img.shape[1] * r))
    new_h = max(1, int(img.shape[0] * r))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded_img[:new_h, :new_w] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


_GRID_CACHE: dict[tuple[int, int, bool], tuple[np.ndarray, np.ndarray]] = {}


def demo_postprocess(outputs: np.ndarray, img_size: tuple[int, int], p6: bool = False) -> np.ndarray:
    """Decode raw YOLOX FPN anchor-free outputs into [cx, cy, w, h, obj_conf, cls_scores...] in input_size coords."""
    key = (img_size[0], img_size[1], p6)
    cached = _GRID_CACHE.get(key)
    if cached is None:
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        for stride in strides:
            hsize = img_size[0] // stride
            wsize = img_size[1] // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2).astype(np.float32)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride, dtype=np.float32))
        cached = (np.concatenate(grids, 1), np.concatenate(expanded_strides, 1))
        _GRID_CACHE[key] = cached

    grids, expanded_strides = cached
    decoded = outputs.copy()
    decoded[..., :2] = (decoded[..., :2] + grids) * expanded_strides
    decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * expanded_strides
    return decoded


def nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> list[int]:
    """Single-class Non-Maximum Suppression in NumPy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    nms_thr: float = 0.45,
    score_thr: float = 0.25,
    classes: Sequence[int] | None = None,
) -> np.ndarray | None:
    """Class-aware NMS returning (N, 6) array of [x1, y1, x2, y2, score, class_id] or None."""
    final_dets = []
    class_indices = classes if classes is not None else range(scores.shape[1])
    for cls_ind in class_indices:
        if cls_ind < 0 or cls_ind >= scores.shape[1]:
            continue
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores >= score_thr
        if not np.any(valid_score_mask):
            continue
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        keep = nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            cls_col = np.full((len(keep), 1), float(cls_ind), dtype=np.float32)
            dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_col], axis=1)
            final_dets.append(dets)
    if not final_dets:
        return None
    return np.concatenate(final_dets, axis=0)


class YOLOXDetector:
    """Drop-in Apache-2.0 person/object detector running on ONNX Runtime and returning `sv.Detections`."""

    def __init__(
        self,
        model_path: str | Path = "yolox_nano.onnx",
        providers: Sequence[str] | None = None,
        nms_thresh: float = 0.45,
        default_imgsz: int = 416,
    ) -> None:
        self.onnx_path = ensure_yolox_onnx(model_path)
        raw_providers = list(providers) if providers else ["CPUExecutionProvider"]
        # Drop CoreMLExecutionProvider for YOLOX as 1 unsupported node causes slow graph partitioning
        filtered_providers = [p for p in raw_providers if p != "CoreMLExecutionProvider"]
        self.providers = filtered_providers or ["CPUExecutionProvider"]
        self.nms_thresh = nms_thresh
        self._sessions: dict[tuple[int, int], ort.InferenceSession] = {}
        self._base_proto = None
        try:
            import onnx

            self._base_proto = onnx.load(str(self.onnx_path))
        except Exception:
            self._base_proto = None

        self.default_hw = self._resolve_input_size(default_imgsz)
        self.session = self._get_session(self.default_hw)
        self.input_name = self.session.get_inputs()[0].name

    def _resolve_input_size(self, imgsz: int | tuple[int, int] | None) -> tuple[int, int]:
        if imgsz is None:
            return (416, 416)
        if isinstance(imgsz, int):
            side = max(32, (imgsz // 32) * 32)
            return (side, side)
        h = max(32, (int(imgsz[0]) // 32) * 32)
        w = max(32, (int(imgsz[1]) // 32) * 32)
        return (h, w)

    def _get_session(self, input_size: tuple[int, int]) -> ort.InferenceSession:
        cached = self._sessions.get(input_size)
        if cached is not None:
            return cached

        h, w = input_size
        if self._base_proto is not None:
            try:
                import onnx

                proto = onnx.ModelProto()
                proto.CopyFrom(self._base_proto)
                inp_dims = proto.graph.input[0].type.tensor_type.shape.dim
                for idx, val in enumerate((1, 3, h, w)):
                    inp_dims[idx].ClearField("dim_param")
                    inp_dims[idx].dim_value = val
                anchors = sum((h // s) * (w // s) for s in (8, 16, 32))
                out_dims = proto.graph.output[0].type.tensor_type.shape.dim
                for idx, val in enumerate((1, anchors, 85)):
                    out_dims[idx].ClearField("dim_param")
                    out_dims[idx].dim_value = val
                del proto.graph.value_info[:]
                proto = onnx.shape_inference.infer_shapes(proto)
                sess = ort.InferenceSession(proto.SerializeToString(), providers=self.providers)
                self._sessions[input_size] = sess
                self.session = sess
                return sess
            except Exception:
                pass

        sess = ort.InferenceSession(str(self.onnx_path), providers=self.providers)
        self._sessions[input_size] = sess
        self.session = sess
        return sess

    def detect_detections(
        self,
        frame: np.ndarray,
        classes: Sequence[int] = (0,),
        conf: float = 0.40,
        imgsz: int | tuple[int, int] = 416,
    ) -> sv.Detections:
        """Run YOLOX inference on `frame` and return a `supervision.Detections` instance."""
        if frame is None or frame.size == 0:
            return sv.Detections.empty()

        input_size = self._resolve_input_size(imgsz)
        sess = self._get_session(input_size)
        blob, ratio = preproc(frame, input_size)
        raw_out = sess.run(None, {self.input_name: blob[None, :, :, :]})[0]
        predictions = demo_postprocess(raw_out, input_size)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nms_thresh, score_thr=conf, classes=classes)
        if dets is None or len(dets) == 0:
            return sv.Detections.empty()

        h, w = frame.shape[:2]
        xyxy = dets[:, :4].astype(np.float32)
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, w)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, h)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, w)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, h)

        return sv.Detections(
            xyxy=xyxy,
            confidence=dets[:, 4].astype(np.float32),
            class_id=dets[:, 5].astype(int),
        )

    def __call__(
        self,
        frame: np.ndarray,
        classes: Sequence[int] = (0,),
        conf: float = 0.40,
        imgsz: int | tuple[int, int] = 416,
    ) -> sv.Detections:
        return self.detect_detections(frame, classes=classes, conf=conf, imgsz=imgsz)
