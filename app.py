"""StaffLens AI: hybrid staff recognition and unique daily customer counting."""

from __future__ import annotations

import base64
import inspect
import json
import os
import queue
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")

import cv2
import numpy as np
import requests
import supervision as sv
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from sqlalchemy import func, select
from ultralytics import YOLO

from database import CameraEvent, DailyVisitor, Employee, FaceEmbedding, SessionLocal, initialize_database

load_dotenv()

CAMERA_ID = os.getenv("CAMERA_ID", "local-webcam-01")
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.40"))
FACE_MODEL = os.getenv("FACE_MODEL", "buffalo_l")
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))
GRAY_ZONE_THRESHOLD = float(os.getenv("FACE_GRAY_ZONE_THRESHOLD", "0.30"))
VISITOR_FACE_THRESHOLD = float(os.getenv("VISITOR_FACE_THRESHOLD", "0.40"))
BODY_MATCH_THRESHOLD = float(os.getenv("BODY_MATCH_THRESHOLD", "0.65"))
CUSTOMER_MIN_DWELL_SECONDS = float(os.getenv("CUSTOMER_MIN_DWELL_SECONDS", "3.5"))
REENTRY_GAP_SECONDS = float(os.getenv("VISITOR_REENTRY_GAP_SECONDS", "30"))
CACHE_TTL = int(os.getenv("IDENTITY_CACHE_TTL_SECONDS", "60"))
EMPLOYEE_REFRESH_SECONDS = max(1, int(os.getenv("EMPLOYEE_REFRESH_SECONDS", "15")))
CUSTOMER_SNAPSHOT_DIR = Path(os.getenv("CUSTOMER_SNAPSHOT_DIR", "customer_snapshots/temp"))


def save_customer_snapshot(crop: np.ndarray, track_id: int) -> str | None:
    """Persist first detected crop before face/customer analysis; never block inference on failure."""
    if crop is None or crop.size == 0:
        return None
    try:
        CUSTOMER_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = CUSTOMER_SNAPSHOT_DIR / f"track_{track_id}_{stamp}.jpg"
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            path.write_bytes(encoded.tobytes())
            return str(path)
    except Exception as exc:
        print(f"Customer snapshot gagal disimpan: {exc}")
    return None

# Optional Smart Fallback Vision AI (OpenAI-compatible /chat/completions endpoint)
VISION_AI_BASE_URL = os.getenv("VISION_AI_BASE_URL", "https://api.openai.com/v1/chat/completions").strip()
VISION_AI_API_KEY = os.getenv("VISION_AI_API_KEY", "").strip()
VISION_AI_MODEL = os.getenv("VISION_AI_MODEL", "gpt-4o-mini").strip()
VISION_AI_TIMEOUT_SECONDS = min(30, max(5, int(os.getenv("VISION_AI_TIMEOUT_SECONDS", "30"))))
SNAPSHOT_RETENTION_SECONDS = max(3600, int(os.getenv("SNAPSHOT_RETENTION_SECONDS", "86400")))
VISION_AI_STAFF_HINT = os.getenv(
    "VISION_AI_STAFF_HINT",
    "Staf klinik QDerma memakai seragam kerja klinik (scrub/tunik medis), jas dokter putih, atau lanyard ID card. "
    "Pengunjung/customer memakai pakaian kasual/bebas.",
).strip()

def cleanup_temp_snapshots() -> None:
    """Best-effort retention cleanup; never blocks/fails the camera loop."""
    try:
        if not CUSTOMER_SNAPSHOT_DIR.exists():
            return
        cutoff = time.time() - SNAPSHOT_RETENTION_SECONDS
        for path in CUSTOMER_SNAPSHOT_DIR.glob("*.jpg"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"Snapshot cleanup skipped: {exc}")


# Level 1 = current quality; higher levels use less CPU but detect smaller faces less reliably.
PERFORMANCE_PRESETS = {
    1: (640, 480, 416, 320, 10),
    2: (640, 480, 320, 320, 15),
    3: (480, 360, 320, 256, 20),
    4: (320, 240, 256, 192, 25),
    5: (320, 240, 192, 160, 30),
}
PERFORMANCE_LEVEL = int(os.getenv("PERFORMANCE_LEVEL", "1"))
if PERFORMANCE_LEVEL not in PERFORMANCE_PRESETS:
    raise ValueError("PERFORMANCE_LEVEL must be a number from 1 to 5")
_preset_width, _preset_height, _preset_imgsz, _preset_face_size, _preset_interval = PERFORMANCE_PRESETS[PERFORMANCE_LEVEL]
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH") or _preset_width)
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT") or _preset_height)
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ") or _preset_imgsz)
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE") or _preset_face_size)
FACE_INTERVAL = int(os.getenv("FACE_DETECTION_INTERVAL") or _preset_interval)
def _resolve_onnx_providers(raw: str) -> list[str]:
    """A2: 'auto' = CoreML (if available) + CPU. Unavailable providers are dropped; CPU is always the fallback."""
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        available = {"CPUExecutionProvider"}
    value = raw.strip()
    if value.lower() == "auto":
        wanted = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        wanted = [item.strip() for item in value.split(",") if item.strip()]
    providers = [item for item in wanted if item in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def _resolve_yolo_device(raw: str) -> str | None:
    """A1: '' = Ultralytics default (CUDA if present else CPU); 'auto' = cuda -> mps -> cpu; or an explicit device."""
    value = raw.strip().lower()
    if not value:
        return None
    if value != "auto":
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _face_allowed_modules(raw: str) -> list[str] | None:
    """B3: only load the InsightFace modules the pipeline uses. 'all' or empty = load every module (baseline)."""
    value = raw.strip().lower()
    if not value or value == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off", ""}


PROVIDERS = _resolve_onnx_providers(os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider"))
YOLO_DEVICE = _resolve_yolo_device(os.getenv("YOLO_DEVICE", ""))
FACE_ALLOWED_MODULES = _face_allowed_modules(os.getenv("FACE_ALLOWED_MODULES", "detection,recognition"))
CAMERA_THREADED = _env_flag("CAMERA_THREADED", "1")
DB_ASYNC_WRITES = _env_flag("DB_ASYNC_WRITES", "1")

_vision_semaphore = threading.Semaphore(1)


def load_yolo_model() -> tuple[YOLO, str]:
    """D8: load YOLO_MODEL; if an exported .onnx file is missing, fall back to the matching .pt weights."""
    path = YOLO_MODEL
    if path.lower().endswith(".onnx") and not os.path.exists(path):
        fallback = path[: -len(".onnx")] + ".pt"
        print(f"YOLO model {path} not found; falling back to {fallback}. See docs/OPTIMASI_PERFORMA.md section 4.")
        path = fallback
    if path.lower().endswith(".onnx"):
        return YOLO(path, task="detect"), path
    return YOLO(path), path


class LatestFrameReader:
    """C4: read the camera on a background thread and always hand out the newest frame (stale frames are dropped)."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._ok = True
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopped:
            ok, frame = self._cap.read()
            with self._cond:
                if not ok:
                    self._ok = False
                    self._cond.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq: int, timeout: float = 10.0) -> tuple[bool, np.ndarray | None, int]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self._ok or self._stopped, timeout=timeout)
            if self._seq != last_seq and self._frame is not None:
                return True, self._frame, self._seq
            return False, None, self._seq

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=2.0)


class BackgroundDbWriter:
    """C6: run non-critical DB writes on one FIFO worker thread so inference does not wait for PostgreSQL."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                print(f"Background DB write failed: {exc}")

    def submit(self, fn, *args, **kwargs) -> None:
        self._queue.put((fn, args, kwargs))

    def close(self, timeout: float = 5.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)


@dataclass(frozen=True)
class CameraConfig:
    video_source: int
    performance_level: int
    camera_width: int | None = None
    camera_height: int | None = None
    customer_min_dwell_seconds: float = 0.5


@dataclass
class VisitorEntry:
    id: int
    visitor_code: str
    short_label: str
    face_vector: np.ndarray | None
    body_vector: np.ndarray | None
    visit_count: int
    last_seen_at: datetime
    last_track_id: int | None
    attributes_summary: str | None = None


def vision_ai_enabled() -> bool:
    flag = os.getenv("VISION_AI_ENABLED", "auto").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(VISION_AI_API_KEY and VISION_AI_BASE_URL)


def local_today() -> date:
    return datetime.now().astimezone().date()


def source(config: CameraConfig | None = None) -> int | str:
    if config is not None:
        return config.video_source
    return int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else os.environ.get("RTSP_URL", VIDEO_SOURCE)


def normalized(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    return arr / (float(np.linalg.norm(arr)) + 1e-12)


def extract_body_signature(crop: np.ndarray) -> np.ndarray | None:
    """Extract an illumination-resilient upper/lower torso HSV color & tone signature for same-day Re-ID."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 40 or w < 20:
        return None

    x1, x2 = int(w * 0.20), int(w * 0.80)
    if x2 <= x1:
        x1, x2 = 0, w
    upper = crop[int(h * 0.16):int(h * 0.54), x1:x2]
    lower = crop[int(h * 0.54):int(h * 0.88), x1:x2]
    if upper.size == 0 or lower.size == 0:
        return None

    features: list[np.ndarray] = []
    for region, weight in ((upper, 1.25), (lower, 0.85)):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hs_hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        part = np.concatenate([hs_hist, v_hist]).astype(np.float32)
        part = part / (float(np.sum(part)) + 1e-6)
        features.append(part * weight)

    aspect = np.array([min(h / max(w, 1), 4.0) * 0.15], dtype=np.float32)
    return normalized(np.concatenate([*features, aspect]))


def blend_vectors(old_vec: np.ndarray | None, new_vec: np.ndarray | None, alpha: float = 0.35) -> np.ndarray | None:
    if new_vec is None:
        return old_vec
    if old_vec is None or old_vec.shape != new_vec.shape:
        return normalized(new_vec)
    return normalized((1.0 - alpha) * old_vec + alpha * new_vec)


def _staff_camera_label(employee: Employee) -> str:
    code = employee.employee_code.upper().removeprefix("EMP-")
    return f"K-{code}-{employee.name}"


def _short_visitor_label(visitor_code: str, visit_count: int = 1) -> str:
    suffix = visitor_code.rsplit("-", 1)[-1]
    base = f"CUST-{suffix}"
    return f"{base} (Kunjungan {visit_count})" if visit_count > 1 else base


def load_known_faces() -> list[tuple[Employee, np.ndarray]]:
    with SessionLocal() as session:
        rows = (
            session.query(Employee, FaceEmbedding)
            .join(FaceEmbedding, FaceEmbedding.employee_id == Employee.id)
            .filter(Employee.active.is_(True), FaceEmbedding.model_name == FACE_MODEL)
            .all()
        )
        return [(employee, normalized(np.asarray(embedding.vector, dtype=np.float32))) for employee, embedding in rows]


def load_today_visitors(target_date: date) -> list[VisitorEntry]:
    with SessionLocal() as session:
        rows = session.scalars(
            select(DailyVisitor)
            .where(DailyVisitor.visit_date == target_date, DailyVisitor.reclassified_employee_id.is_(None))
            .order_by(DailyVisitor.id.asc())
        ).all()
        entries: list[VisitorEntry] = []
        for row in rows:
            f_vec = normalized(np.asarray(row.face_vector, dtype=np.float32)) if row.face_vector else None
            b_vec = normalized(np.asarray(row.body_vector, dtype=np.float32)) if row.body_vector else None
            entries.append(
                VisitorEntry(
                    id=row.id,
                    visitor_code=row.visitor_code,
                    short_label=_short_visitor_label(row.visitor_code, row.visit_count),
                    face_vector=f_vec,
                    body_vector=b_vec,
                    visit_count=row.visit_count,
                    last_seen_at=row.last_seen_at,
                    last_track_id=row.last_track_id,
                    attributes_summary=row.attributes_summary,
                )
            )
        return entries


def match_employee_best(
    embedding: np.ndarray,
    known: list[tuple[Employee, np.ndarray]],
) -> tuple[Employee, float] | None:
    probe = normalized(embedding)
    best: tuple[Employee, float] | None = None
    for employee, vector in known:
        score = float(np.dot(probe, vector))
        if best is None or score > best[1]:
            best = (employee, score)
    return best


def match_employee(embedding: np.ndarray, known: list[tuple[Employee, np.ndarray]]) -> tuple[Employee, float] | None:
    best = match_employee_best(embedding, known)
    return best if best and best[1] >= MATCH_THRESHOLD else None


def match_staff_body(
    body_vec: np.ndarray | None,
    staff_body_cache: dict[int, tuple[Employee, np.ndarray, float]],
    now: float,
) -> tuple[Employee, float] | None:
    if body_vec is None:
        return None
    best: tuple[Employee, float] | None = None
    for emp, vec, seen_at in list(staff_body_cache.values()):
        if now - seen_at > 4 * 3600:
            continue
        score = float(np.dot(body_vec, vec))
        if best is None or score > best[1]:
            best = (emp, score)
    return best if best and best[1] >= max(BODY_MATCH_THRESHOLD + 0.04, 0.88) else None


def match_existing_visitor(
    face_vec: np.ndarray | None,
    body_vec: np.ndarray | None,
    visitors: list[VisitorEntry],
) -> tuple[VisitorEntry, float, str] | None:
    """Match a non-employee against today's visitors so returning visitors on the same day are not double-counted."""
    if face_vec is not None:
        best_face: tuple[VisitorEntry, float] | None = None
        for visitor in visitors:
            if visitor.face_vector is None:
                continue
            score = float(np.dot(face_vec, visitor.face_vector))
            if best_face is None or score > best_face[1]:
                best_face = (visitor, score)
        if best_face and best_face[1] >= VISITOR_FACE_THRESHOLD:
            return best_face[0], best_face[1], "face"

    if body_vec is not None:
        best_body: tuple[VisitorEntry, float] | None = None
        for visitor in visitors:
            if visitor.body_vector is None:
                continue
            # If both have clear face vectors that strongly disagree, do not merge by clothing alone.
            if face_vec is not None and visitor.face_vector is not None:
                if float(np.dot(face_vec, visitor.face_vector)) < 0.20:
                    continue
            score = float(np.dot(body_vec, visitor.body_vector))
            if best_body is None or score > best_body[1]:
                best_body = (visitor, score)
        if best_body and best_body[1] >= BODY_MATCH_THRESHOLD:
            return best_body[0], best_body[1], "body"

    return None


def log_identity(track_id: int, employee: Employee, confidence: float, match_source: str = "face") -> None:
    now_utc = datetime.now(timezone.utc)
    with SessionLocal() as session:
        # Prevent duplicate event spam if the same employee was just logged on the same track within 2 minutes
        recent = session.scalar(
            select(CameraEvent)
            .where(
                CameraEvent.camera_id == CAMERA_ID,
                CameraEvent.employee_id == employee.id,
                CameraEvent.track_id == track_id,
                CameraEvent.created_at >= now_utc - timedelta(seconds=120),
            )
            .limit(1)
        )
        if recent is not None:
            return
        session.add(
            CameraEvent(
                camera_id=CAMERA_ID,
                track_id=track_id,
                employee_id=employee.id,
                label=f"{employee.role} - {employee.name}",
                confidence=confidence,
                event_type="identity_locked",
                metadata_json={
                    "source": "webcam",
                    "kind": "staff",
                    "match_source": match_source,
                    "locked_at": now_utc.isoformat(),
                },
            )
        )
        session.commit()


def register_or_update_visitor(
    track_id: int,
    target_date: date,
    face_vec: np.ndarray | None,
    body_vec: np.ndarray | None,
    visitors: list[VisitorEntry],
    attributes_summary: str | None = None,
) -> tuple[VisitorEntry, bool]:
    """Return (VisitorEntry, is_new_unique_customer). Guarantees 1 unique customer per day even on multiple visits."""
    now_utc = datetime.now(timezone.utc)
    matched = match_existing_visitor(face_vec, body_vec, visitors)

    with SessionLocal() as session:
        if matched is not None:
            visitor_entry, score, match_mode = matched
            db_row = session.get(DailyVisitor, visitor_entry.id)
            if db_row is not None and db_row.reclassified_employee_id is None:
                gap_seconds = (now_utc - db_row.last_seen_at).total_seconds() if db_row.last_seen_at else 0.0
                is_reentry = db_row.last_track_id != track_id and gap_seconds >= REENTRY_GAP_SECONDS
                if is_reentry:
                    db_row.visit_count += 1
                db_row.last_seen_at = now_utc
                db_row.last_track_id = track_id
                db_row.confidence = max(float(db_row.confidence or 0.0), score)
                merged_face = blend_vectors(visitor_entry.face_vector, face_vec)
                merged_body = blend_vectors(visitor_entry.body_vector, body_vec)
                if merged_face is not None:
                    db_row.face_vector = merged_face.tolist()
                if merged_body is not None:
                    db_row.body_vector = merged_body.tolist()
                if attributes_summary and not db_row.attributes_summary:
                    db_row.attributes_summary = attributes_summary

                visitor_entry.face_vector = merged_face
                visitor_entry.body_vector = merged_body
                visitor_entry.visit_count = db_row.visit_count
                visitor_entry.last_seen_at = now_utc
                visitor_entry.last_track_id = track_id
                visitor_entry.short_label = _short_visitor_label(db_row.visitor_code, db_row.visit_count)
                if attributes_summary and not visitor_entry.attributes_summary:
                    visitor_entry.attributes_summary = attributes_summary

                if is_reentry:
                    session.add(
                        CameraEvent(
                            camera_id=CAMERA_ID,
                            track_id=track_id,
                            employee_id=None,
                            label=visitor_entry.short_label,
                            confidence=score,
                            event_type="customer_return",
                            metadata_json={
                                "source": "webcam",
                                "kind": "customer",
                                "visitor_id": db_row.id,
                                "visitor_code": db_row.visitor_code,
                                "visit_count": db_row.visit_count,
                                "match_mode": match_mode,
                            },
                        )
                    )
                session.commit()
                return visitor_entry, False

        # Create a new unique customer for today
        seq = (
            session.scalar(
                select(func.count(DailyVisitor.id)).where(DailyVisitor.visit_date == target_date)
            )
            or 0
        ) + 1
        date_str = target_date.strftime("%Y%m%d")
        visitor_code = f"CUST-{date_str}-{seq:03d}"
        while session.scalar(select(DailyVisitor.id).where(DailyVisitor.visitor_code == visitor_code)) is not None:
            seq += 1
            visitor_code = f"CUST-{date_str}-{seq:03d}"

        confidence = 0.85 if face_vec is not None else 0.70
        new_row = DailyVisitor(
            visit_date=target_date,
            visitor_code=visitor_code,
            face_vector=face_vec.tolist() if face_vec is not None else None,
            body_vector=body_vec.tolist() if body_vec is not None else None,
            attributes_summary=attributes_summary or ("Wajah + Pakaian" if face_vec is not None else "Pola Pakaian"),
            confidence=confidence,
            first_seen_at=now_utc,
            last_seen_at=now_utc,
            visit_count=1,
            last_track_id=track_id,
        )
        session.add(new_row)
        session.flush()

        entry = VisitorEntry(
            id=new_row.id,
            visitor_code=visitor_code,
            short_label=_short_visitor_label(visitor_code, 1),
            face_vector=face_vec,
            body_vector=body_vec,
            visit_count=1,
            last_seen_at=now_utc,
            last_track_id=track_id,
            attributes_summary=new_row.attributes_summary,
        )
        visitors.append(entry)

        session.add(
            CameraEvent(
                camera_id=CAMERA_ID,
                track_id=track_id,
                employee_id=None,
                label=entry.short_label,
                confidence=confidence,
                event_type="customer_new",
                metadata_json={
                    "source": "webcam",
                    "kind": "customer",
                    "visitor_id": new_row.id,
                    "visitor_code": visitor_code,
                    "visit_count": 1,
                    "match_mode": "face+body" if face_vec is not None else "body",
                },
            )
        )
        session.commit()
        return entry, True


def update_visitor_attributes(visitor_id: int, attributes_summary: str) -> None:
    with SessionLocal() as session:
        row = session.get(DailyVisitor, visitor_id)
        if row is not None:
            row.attributes_summary = attributes_summary
            session.commit()


def update_visitor_face(visitor_id: int, face_vector: list[float]) -> None:
    with SessionLocal() as session:
        row = session.get(DailyVisitor, visitor_id)
        if row is not None:
            row.face_vector = face_vector
            row.attributes_summary = "Wajah + Pakaian"
            session.commit()


def reclassify_visitor_to_employee(
    visitor_id: int | None,
    employee: Employee | None,
    track_id: int,
    score: float,
    visitors: list[VisitorEntry],
) -> None:
    """Rollback a provisional customer record ONLY if this exact track_id was previously registered as that visitor."""
    if visitor_id is None:
        return

    visitors[:] = [v for v in visitors if v.id != visitor_id]
    with SessionLocal() as session:
        row = session.get(DailyVisitor, visitor_id)
        visitor_code = row.visitor_code if row is not None else None
        if employee is not None and visitor_code:
            import shutil
            from pathlib import Path

            customer_dir = Path(__file__).parent / "db_customer" / visitor_code
            shutil.rmtree(customer_dir, ignore_errors=True)
        if row is not None:
            if employee is not None:
                row.reclassified_employee_id = employee.id
            else:
                session.delete(row)

        events = session.scalars(
            select(CameraEvent).where(
                CameraEvent.camera_id == CAMERA_ID,
                CameraEvent.track_id == track_id,
                CameraEvent.event_type.in_(["customer_new", "customer_return"]),
            )
        ).all()
        for ev in events:
            if employee is not None:
                ev.employee_id = employee.id
                ev.label = f"{employee.role} - {employee.name}"
                ev.confidence = score
                ev.event_type = "identity_locked"
                ev.metadata_json = {
                    **(ev.metadata_json or {}),
                    "kind": "staff",
                    "reclassified_from_visitor_id": visitor_id,
                }
            else:
                session.delete(ev)
        session.commit()


def verify_crop_with_vision_ai(crop: np.ndarray) -> dict[str, object] | None:
    """Send 1 snapshot crop to Vision AI endpoint to verify staff uniform vs customer clothing."""
    if not vision_ai_enabled():
        return None
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    b64_image = base64.b64encode(encoded.tobytes()).decode("ascii")
    prompt = (
        f"{VISION_AI_STAFF_HINT}\n"
        "Periksa gambar orang ini dan jawab HANYA dalam format JSON valid tanpa markdown:\n"
        '{"role": "staff" atau "customer", "confidence": 0.0-1.0, "attributes": "ciri singkat pakaian maks 6 kata"}'
    )
    headers = {"Authorization": f"Bearer {VISION_AI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": VISION_AI_MODEL,
        "temperature": 0.1,
        "max_tokens": 120,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            }
        ],
    }
    try:
        resp = requests.post(
            VISION_AI_BASE_URL, headers=headers, json=payload, timeout=VISION_AI_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`").removeprefix("json").strip()
        data = json.loads(content)
        if isinstance(data, dict) and data.get("role") in {"staff", "customer"}:
            return data
    except requests.Timeout:
        print(f"Vision AI fallback timed out after {VISION_AI_TIMEOUT_SECONDS}s")
    except Exception as exc:
        print(f"Vision AI fallback skipped: {exc}")
    return None


def main(
    stop_event=None,
    on_frame=None,
    on_customer_snapshot=None,
    show_window: bool = True,
    config: CameraConfig | None = None,
) -> None:
    initialize_database()
    width, height, imgsz, face_size, interval = (
        PERFORMANCE_PRESETS[config.performance_level]
        if config
        else (CAMERA_WIDTH, CAMERA_HEIGHT, YOLO_IMGSZ, FACE_DET_SIZE, FACE_INTERVAL)
    )
    customer_min_dwell_seconds = config.customer_min_dwell_seconds if config else CUSTOMER_MIN_DWELL_SECONDS
    if config:
        width = config.camera_width or width
        height = config.camera_height or height

    current_date = local_today()
    known = load_known_faces()
    today_visitors = load_today_visitors(current_date)
    print(
        f"Loaded {len(known)} employee embeddings and {len(today_visitors)} unique visitor(s) for {current_date}. "
        "Press q to quit."
    )

    model, yolo_path = load_yolo_model()
    yolo_device = YOLO_DEVICE
    if yolo_device == "mps" and yolo_path.lower().endswith(".onnx"):
        yolo_device = "cpu"
    yolo_kwargs: dict[str, object] = {"device": yolo_device} if yolo_device else {}
    tracker = sv.ByteTrack()
    face_app = FaceAnalysis(name=FACE_MODEL, providers=PROVIDERS, allowed_modules=FACE_ALLOWED_MODULES)
    face_app.prepare(ctx_id=0, det_size=(face_size, face_size))
    print(
        f"Runtime: YOLO={yolo_path} device={yolo_device or 'default'} | FACE_MODEL={FACE_MODEL} "
        f"modules={sorted(face_app.models)} providers={PROVIDERS} | threaded_capture={CAMERA_THREADED} "
        f"async_db={DB_ASYNC_WRITES}"
    )

    if stop_event is not None and stop_event.is_set():
        return
    cap = cv2.VideoCapture(source(config))
    if not cap.isOpened():
        raise RuntimeError("Camera unavailable. Grant Camera permission to Terminal/VS Code, then retry.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    reader = LatestFrameReader(cap) if CAMERA_THREADED else None
    reader_seq = 0
    db_writer = BackgroundDbWriter() if DB_ASYNC_WRITES else None
    snapshot_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot-writer")
    cleanup_temp_snapshots()

    def db_write(fn, *args, **kwargs) -> None:
        if db_writer is not None:
            db_writer.submit(fn, *args, **kwargs)
        else:
            fn(*args, **kwargs)

    # track_id -> state dict
    cache: dict[int, dict[str, object]] = {}
    # employee_id -> (Employee, body_vector, timestamp)
    staff_body_cache: dict[int, tuple[Employee, np.ndarray, float]] = {}
    vision_results: dict[int, dict[str, object]] = {}
    vision_lock = threading.Lock()

    accepts_live_counts = False
    if on_frame is not None:
        try:
            accepts_live_counts = len(inspect.signature(on_frame).parameters) >= 3
        except (TypeError, ValueError):
            accepts_live_counts = False

    frame_number = 0
    last_refresh = time.monotonic()
    last_frame = last_refresh
    try:
        while stop_event is None or not stop_event.is_set():
            if reader is not None:
                ok, frame, reader_seq = reader.read(reader_seq)
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_number += 1
            now = time.monotonic()
            fps = 1 / max(now - last_frame, 1e-6)
            last_frame = now

            # Check date rollover (midnight reset for daily visitors) & periodic DB refresh
            today_check = local_today()
            if today_check != current_date:
                current_date = today_check
                today_visitors = load_today_visitors(current_date)
                cache.clear()
                staff_body_cache.clear()
            elif now - last_refresh >= EMPLOYEE_REFRESH_SECONDS:
                try:
                    known = load_known_faces()
                    today_visitors = load_today_visitors(current_date)
                    names = {employee.id: _staff_camera_label(employee) for employee, _ in known}
                    for track_id, entry in list(cache.items()):
                        if entry.get("kind") == "staff" and entry.get("employee_id") in names:
                            entry["label"] = names[entry["employee_id"]]
                except Exception as exc:
                    print(f"Database refresh failed; keeping previous data: {exc}")
                last_refresh = now

            # Apply any completed async Vision AI fallback results
            if vision_results:
                with vision_lock:
                    completed_vision = list(vision_results.items())
                    vision_results.clear()
                for v_track_id, v_data in completed_vision:
                    t_entry = cache.get(v_track_id)
                    if not t_entry or t_entry.get("kind") == "staff":
                        continue
                    v_role = str(v_data.get("role", ""))
                    v_conf = float(v_data.get("confidence") or 0.0)
                    v_attr = str(v_data.get("attributes") or "").strip()
                    if v_role == "staff" and v_conf >= 0.65:
                        candidate_emp: Employee | None = t_entry.get("gray_employee")  # type: ignore[assignment]
                        if t_entry.get("visitor_id") is not None:
                            reclassify_visitor_to_employee(
                                int(t_entry["visitor_id"]),
                                candidate_emp,
                                v_track_id,
                                v_conf,
                                today_visitors,
                            )
                        if candidate_emp is not None:
                            t_entry.update(
                                {
                                    "kind": "staff",
                                    "employee_id": candidate_emp.id,
                                    "visitor_id": None,
                                    "label": _staff_camera_label(candidate_emp),
                                    "last_seen": now,
                                }
                            )
                            db_write(log_identity, v_track_id, candidate_emp, v_conf, match_source="vision_ai+face")
                        else:
                            t_entry.update(
                                {
                                    "kind": "staff",
                                    "employee_id": None,
                                    "visitor_id": None,
                                    "label": "Karyawan AI (belum teridentifikasi)",
                                    "last_seen": now,
                                }
                            )
                    elif v_role == "customer" and v_attr and t_entry.get("visitor_id") is not None:
                        vis_id = int(t_entry["visitor_id"])
                        db_write(update_visitor_attributes, vis_id, v_attr[:255])
                        for v_item in today_visitors:
                            if v_item.id == vis_id:
                                v_item.attributes_summary = v_attr[:255]

            # Evict stale tracks that have not been seen for CACHE_TTL seconds
            for stale_id, stale_entry in list(cache.items()):
                if now - float(stale_entry.get("last_seen", now)) > CACHE_TTL:
                    cache.pop(stale_id, None)

            result = model(frame, classes=[0], conf=YOLO_CONFIDENCE, imgsz=imgsz, verbose=False, **yolo_kwargs)[0]
            detections = tracker.update_with_detections(sv.Detections.from_ultralytics(result))
            face_checked = False

            live_staff = 0
            live_customers = 0
            live_observing = 0

            for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
                if tracker_id is None:
                    continue
                track_id = int(tracker_id)
                x1, y1, x2, y2 = map(int, xyxy)
                crop = frame[max(y1, 0):min(y2, frame.shape[0]), max(x1, 0):min(x2, frame.shape[1])]

                entry = cache.get(track_id)
                if entry is None:
                    snapshot_job = None
                    if crop.size:
                        snapshot_job = snapshot_writer.submit(save_customer_snapshot, crop.copy(), track_id)
                    entry = {
                        "kind": "observing",
                        "snapshot_job": snapshot_job,
                        "first_seen": now,
                        "last_seen": now,
                        "label": f"Observasi... #{track_id}",
                        "employee_id": None,
                        "visitor_id": None,
                        "face_vector": None,
                        "body_vector": extract_body_signature(crop),
                        "gray_employee": None,
                        "gray_score": 0.0,
                        "vision_queried": False,
                    }
                    cache[track_id] = entry
                else:
                    entry["last_seen"] = now
                    if frame_number % 5 == 0 and crop.size:
                        b_sig = extract_body_signature(crop)
                        if b_sig is not None:
                            entry["body_vector"] = blend_vectors(entry.get("body_vector"), b_sig)  # type: ignore[arg-type]

                # Run selective face recognition on non-staff tracks at configured frame interval
                should_check_face = (
                    entry.get("kind") != "staff"
                    and not face_checked
                    and crop.size > 0
                    and (frame_number % max(interval, 1) == 0 or entry.get("face_checked_once") is not True)
                )
                if should_check_face:
                    face_checked = True
                    entry["face_checked_once"] = True
                    faces = face_app.get(crop)
                    if faces:
                        f_vec = normalized(np.asarray(faces[0].embedding, dtype=np.float32))
                        entry["face_vector"] = blend_vectors(entry.get("face_vector"), f_vec)  # type: ignore[arg-type]
                        best_emp = match_employee_best(f_vec, known)
                        if best_emp and best_emp[1] >= MATCH_THRESHOLD:
                            employee, score = best_emp
                            # If this exact track was provisionally counted as customer before turning around, roll it back!
                            if entry.get("visitor_id") is not None:
                                reclassify_visitor_to_employee(
                                    int(entry["visitor_id"]),
                                    employee,
                                    track_id,
                                    score,
                                    today_visitors,
                                )
                            entry.update(
                                {
                                    "kind": "staff",
                                    "employee_id": employee.id,
                                    "visitor_id": None,
                                    "label": _staff_camera_label(employee),
                                }
                            )
                            if entry.get("body_vector") is not None:
                                staff_body_cache[employee.id] = (employee, entry["body_vector"], now)  # type: ignore[arg-type]
                            db_write(log_identity, track_id, employee, score, match_source="face")
                        elif best_emp and best_emp[1] >= GRAY_ZONE_THRESHOLD:
                            entry["gray_employee"] = best_emp[0]
                            entry["gray_score"] = best_emp[1]

                # Refresh staff body signature when employee is actively tracked
                if entry.get("kind") == "staff" and entry.get("employee_id") is not None and entry.get("body_vector") is not None:
                    for emp_obj, _ in known:
                        if emp_obj.id == entry["employee_id"]:
                            staff_body_cache[emp_obj.id] = (emp_obj, entry["body_vector"], now)  # type: ignore[arg-type]
                            break

                # After grace period, resolve observing track into either Staff (via body/Vision AI) or Unique Daily Customer
                dwell = now - float(entry.get("first_seen", now))
                if entry.get("kind") == "observing" and dwell >= customer_min_dwell_seconds:
                    staff_body_match = match_staff_body(entry.get("body_vector"), staff_body_cache, now)  # type: ignore[arg-type]
                    if staff_body_match is not None and entry.get("face_vector") is None:
                        emp_matched, b_score = staff_body_match
                        entry.update(
                            {
                                "kind": "staff",
                                "employee_id": emp_matched.id,
                                "visitor_id": None,
                                "label": _staff_camera_label(emp_matched),
                            }
                        )
                        db_write(log_identity, track_id, emp_matched, b_score, match_source="staff_body_reid")
                    else:
                        visitor_entry, _ = register_or_update_visitor(
                            track_id=track_id,
                            target_date=current_date,
                            face_vec=entry.get("face_vector"),  # type: ignore[arg-type]
                            body_vec=entry.get("body_vector"),  # type: ignore[arg-type]
                            visitors=today_visitors,
                        )
                        entry.update(
                            {
                                "kind": "customer",
                                "visitor_id": visitor_entry.id,
                                "label": visitor_entry.short_label,
                            }
                        )

                        # Trigger non-blocking Vision AI Fallback if in gray zone or face was never seen
                        if (
                            vision_ai_enabled()
                            and not entry.get("vision_queried")
                            and (entry.get("face_vector") is None or entry.get("gray_employee") is not None)
                            and crop.size > 0
                            and _vision_semaphore.acquire(blocking=False)
                        ):
                            entry["vision_queried"] = True
                            crop_copy = crop.copy()

                            def _bg_vision(tid: int, img_crop: np.ndarray) -> None:
                                try:
                                    res = verify_crop_with_vision_ai(img_crop)
                                    if res:
                                        with vision_lock:
                                            vision_results[tid] = res
                                finally:
                                    _vision_semaphore.release()

                            threading.Thread(target=_bg_vision, args=(track_id, crop_copy), daemon=True).start()

                # Keep existing customer's face/body signature updated if their face becomes visible later
                elif entry.get("kind") == "customer" and entry.get("visitor_id") is not None:
                    vis_id = int(entry["visitor_id"])
                    if frame_number % 30 == 0:
                        for v_item in today_visitors:
                            if v_item.id == vis_id:
                                if entry.get("face_vector") is not None and v_item.face_vector is None:
                                    v_item.face_vector = entry["face_vector"]  # type: ignore[assignment]
                                    db_write(update_visitor_face, vis_id, v_item.face_vector.tolist())
                                break
                    # Sample a track crop every 30 inference frames; server keeps only the 10 sharpest.
                    if on_customer_snapshot is not None and crop.size and frame_number % 30 == 0:
                        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
                        quality = min(100.0, sharpness ** 0.5)
                        if quality >= 2.0:
                            visitor_code = next((v.visitor_code for v in today_visitors if v.id == vis_id), None)
                            if visitor_code:
                                try:
                                    on_customer_snapshot(visitor_code, crop.copy(), quality)
                                except Exception as exc:
                                    print(f"Customer snapshot skipped: {exc}")

                kind = str(entry.get("kind", "observing"))
                if kind == "staff":
                    live_staff += 1
                    box_color = (75, 175, 35)  # Green for active staff
                elif kind == "customer":
                    live_customers += 1
                    box_color = (180, 125, 25)  # Teal-blue for unique customer
                else:
                    live_observing += 1
                    box_color = (0, 155, 245)  # Amber while observing

                label = str(entry.get("label", f"Observasi... #{track_id}"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

            if on_frame is not None:
                if accepts_live_counts:
                    on_frame(
                        frame,
                        fps,
                        {"staff": live_staff, "customers": live_customers, "observing": live_observing},
                    )
                else:
                    on_frame(frame, fps)
            if show_window:
                cv2.imshow("StaffLens AI — q: quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if reader is not None:
            reader.stop()
        cap.release()
        if db_writer is not None:
            db_writer.close()
        snapshot_writer.shutdown(wait=False, cancel_futures=True)
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
