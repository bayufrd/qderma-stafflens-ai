"""Password-protected local dashboard API, employee photo registration, and one-camera worker."""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import platform
import re
import secrets
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import satusehat
from database import (
    AuditLog,
    CameraEvent,
    CameraSettings,
    DailyVisitor,
    Employee,
    FaceEmbedding,
    SatuSehatSettings,
    SessionLocal,
    initialize_database,
)

load_dotenv()

COOKIE_NAME = "stafflens_session"
SESSION_TTL = 8 * 60 * 60
ROOT_DIR = Path(__file__).parent
DIST = ROOT_DIR / "frontend" / "dist"
DB_KARYAWAN_DIR = ROOT_DIR / "db_karyawan"
DB_CUSTOMER_DIR = ROOT_DIR / "db_customer"
CUSTOMER_SNAPSHOT_LIMIT = 10
_CUSTOMER_SNAPSHOT_LOCK = threading.Lock()
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# C5: dashboard preview throttle/resize (baseline: 8 fps, full-size frame, JPEG 65). See docs/OPTIMASI_PERFORMA.md.
PREVIEW_MAX_FPS = max(1.0, float(os.getenv("PREVIEW_MAX_FPS", "15")))
PREVIEW_MAX_WIDTH = max(0, int(os.getenv("PREVIEW_MAX_WIDTH", "640")))
PREVIEW_JPEG_QUALITY = min(100, max(10, int(os.getenv("PREVIEW_JPEG_QUALITY", "65"))))
VISION_AI_TIMEOUT_SECONDS = min(300, max(120, int(os.getenv("VISION_AI_TIMEOUT_SECONDS", "120"))))


def _face_model_name() -> str:
    return os.getenv("FACE_MODEL", "buffalo_l").strip() or "buffalo_l"


def _face_allowed_modules() -> list[str] | None:
    value = os.getenv("FACE_ALLOWED_MODULES", "detection,recognition").strip().lower()
    if not value or value == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]

app = FastAPI(title="StaffLens AI", docs_url=None, redoc_url=None, openapi_url=None)


_QUIET_POLL_PATHS = (
    '"GET /api/auth/session',
    '"GET /api/camera/frame',
    '"GET /api/camera/status',
    '"GET /api/dashboard',
    '"GET /api/overview',
    '"GET /api/visitors',
    '"GET /api/events',
    '"GET /api/employees ',
    '"GET /api/system/resources',
)


class _SuppressFramePollFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        if "/api/camera/frame" in msg or "/api/camera/ws" in msg:
            return False
        if msg in ("connection open", "connection closed"):
            return False
        if msg.endswith(" 200") and any(p in msg for p in _QUIET_POLL_PATHS):
            return False
        return True


_log_filter = _SuppressFramePollFilter()


def _attach_log_filter() -> None:
    for logger_name in ("uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(logger_name)
        if not any(isinstance(f, _SuppressFramePollFilter) for f in lg.filters):
            lg.addFilter(_log_filter)


_attach_log_filter()


@app.on_event("startup")
def _install_log_filter() -> None:
    _attach_log_filter()
    _check_models_on_startup()
    _start_vision_ai_check()


# One process/one camera. Run uvicorn with a single worker; no public network binding.
_state_lock = threading.Lock()
_frame_ready = threading.Condition(_state_lock)
_stop = threading.Event()
_generation = int(time.time() * 1000)
_frame_version = 0
_worker: threading.Thread | None = None
_session_tokens: dict[str, float] = {}
_login_attempts: dict[str, tuple[int, float]] = {}
_frame: bytes | None = None
_fps = 0.0
_live_counts: dict[str, int] = {"staff": 0, "customers": 0, "observing": 0}
_error: str | None = None
_active_config: dict | None = None
_apply_lock = threading.Lock()
_resource_cache: tuple[float, dict] = (0, {})
_resource_lock = threading.Lock()
_CPU_MODEL_NAME: str | None = None
_face_lock = threading.Lock()
_face_analyzer = None
_restarting = False
_db_ready = False
_db_init_lock = threading.Lock()
_vision_status_lock = threading.Lock()
_vision_runtime: dict[str, object] = {
    "status": "checking",
    "last_checked_at": None,
    "message": "Pemeriksaan koneksi belum selesai.",
}
_vision_check_thread: threading.Thread | None = None
_ws_waiters: dict[asyncio.Event, tuple[asyncio.AbstractEventLoop, dict[str, object]]] = {}


def _ensure_db() -> None:
    global _db_ready
    if _db_ready:
        return
    with _db_init_lock:
        if not _db_ready:
            initialize_database()
            _db_ready = True


def _notify_ws_waiters(frame_only: bool = False) -> None:
    # Caller must hold _state_lock / _frame_ready.
    dead: list[asyncio.Event] = []
    for ev, (loop, state) in list(_ws_waiters.items()):
        if frame_only and not state.get("subscribe_frames", False):
            continue
        try:
            loop.call_soon_threadsafe(ev.set)
        except RuntimeError:
            dead.append(ev)
    for ev in dead:
        _ws_waiters.pop(ev, None)


def _get_cpu_model_name() -> str:
    global _CPU_MODEL_NAME
    if _CPU_MODEL_NAME is not None:
        return _CPU_MODEL_NAME
    name = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            name = (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=1).strip()
                or name
            )
        except (OSError, subprocess.SubprocessError):
            pass
    _CPU_MODEL_NAME = name
    return name


_get_cpu_model_name()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return cleaned or "pegawai"


def _resolve_employee_folder(employee_code: str, employee_name: str) -> Path:
    """Find or assign the employee's photo folder inside db_karyawan/."""
    DB_KARYAWAN_DIR.mkdir(parents=True, exist_ok=True)
    name_slug = _slugify(employee_name)
    code_slug = _slugify(employee_code)

    candidates = [
        DB_KARYAWAN_DIR / name_slug,
        DB_KARYAWAN_DIR / code_slug,
    ]
    # Support legacy initial folder contoh_andi for EMP-001 (Bayu)
    if employee_code.upper() == "EMP-001" and not (DB_KARYAWAN_DIR / name_slug).exists():
        legacy = DB_KARYAWAN_DIR / "contoh_andi"
        if legacy.exists():
            try:
                shutil.copytree(legacy, DB_KARYAWAN_DIR / name_slug, dirs_exist_ok=True)
            except OSError:
                candidates.insert(0, legacy)

    for folder in candidates:
        if folder.exists() and folder.is_dir():
            return folder
    return DB_KARYAWAN_DIR / name_slug


def _check_models_on_startup() -> None:
    """Verify critical models exist; warn if missing (non-blocking)."""
    import logging
    
    logger = logging.getLogger("uvicorn.error")
    home = Path.home()
    face_model = _face_model_name()
    insightface_path = home / ".insightface" / "models" / face_model
    
    # Check InsightFace model
    if not insightface_path.exists() or not any(insightface_path.glob("*.onnx")):
        logger.warning(
            f"⚠️  InsightFace model '{face_model}' not found at {insightface_path}. "
            f"It will be downloaded (~140-250 MB) on first camera start. "
            f"To pre-download: python setup_models.py"
        )
    
    # Check YOLO model
    yolo_model = os.getenv("YOLO_MODEL", "yolov8n.pt").strip()
    yolo_path = Path(yolo_model)
    if not yolo_path.exists() and not yolo_path.is_absolute():
        # Try in current directory
        yolo_path = Path.cwd() / yolo_model
    
    if not yolo_path.exists():
        logger.warning(
            f"⚠️  YOLO model '{yolo_model}' not found. "
            f"It will be downloaded (~6 MB) on first camera start. "
            f"To pre-download: python setup_models.py"
        )


def _list_folder_photos(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
    )


def _get_face_analyzer():
    global _face_analyzer
    if _face_analyzer is None:
        from insightface.app import FaceAnalysis

        analyzer = FaceAnalysis(
            name=_face_model_name(), providers=["CPUExecutionProvider"], allowed_modules=_face_allowed_modules()
        )
        analyzer.prepare(ctx_id=0, det_size=(640, 640))
        _face_analyzer = analyzer
    return _face_analyzer


def _extract_vectors_from_folder(folder: Path) -> tuple[list[list[float]], list[str], list[dict[str, str]]]:
    photos = _list_folder_photos(folder)
    if not photos:
        return [], [], []

    with _face_lock:
        face_app = _get_face_analyzer()
        vectors: list[list[float]] = []
        accepted: list[str] = []
        skipped: list[dict[str, str]] = []
        for photo in photos:
            image = cv2.imread(str(photo))
            if image is None:
                skipped.append({"file": photo.name, "reason": "File gambar tidak dapat dibaca"})
                continue
            faces = face_app.get(image)
            if len(faces) != 1:
                reason = "Wajah tidak terdeteksi" if len(faces) == 0 else f"Terdeteksi {len(faces)} wajah (harus 1 wajah)"
                skipped.append({"file": photo.name, "reason": reason})
                continue
            vectors.append(np.asarray(faces[0].embedding, dtype=np.float32).tolist())
            accepted.append(photo.name)
        return vectors, accepted, skipped


def _decode_data_url(data_url: str) -> bytes:
    raw = data_url.strip()
    if "," in raw and raw.startswith("data:"):
        _, encoded = raw.split(",", 1)
    else:
        encoded = raw
    try:
        return base64.b64decode(encoded, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Format gambar unggahan tidak valid") from exc


class LoginBody(BaseModel):
    password: str


class CameraSettingsBody(BaseModel):
    video_source: int = Field(ge=0, le=8)
    performance_level: int = Field(ge=1, le=5)
    camera_width: int | None = Field(default=None, ge=160, le=1920)
    camera_height: int | None = Field(default=None, ge=120, le=1080)
    customer_min_dwell_seconds: float = Field(default=0.5, ge=0.1, le=10)


class PhotoUploadItem(BaseModel):
    filename: str
    data_url: str


class EmployeeSaveBody(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    role: str = Field(default="Karyawan", min_length=1, max_length=100)
    active: bool = True
    nik: str | None = Field(default=None, max_length=32)  # None = keep, "" = clear, 16 digits = set
    new_photos: list[PhotoUploadItem] = Field(default_factory=list)
    delete_photos: list[str] = Field(default_factory=list)
    reprocess_embeddings: bool = False


class SatuSehatSettingsBody(BaseModel):
    env: Literal["dev", "staging", "production"]
    client_id: str = Field(default="", max_length=255)
    organization_id: str = Field(default="", max_length=64)
    client_secret: str | None = Field(default=None, max_length=512)  # None/"" = keep the stored secret
    location_name: str = Field(default="", max_length=255)


def _today_bounds_utc() -> tuple[datetime, datetime]:
    now_local = datetime.now().astimezone()
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_local.astimezone(timezone.utc), now_local.astimezone(timezone.utc)


def _vision_ai_config() -> tuple[bool, str, str, str]:
    flag = os.getenv("VISION_AI_ENABLED", "auto").strip().lower()
    api_key = os.getenv("VISION_AI_API_KEY", "").strip()
    base_url = os.getenv(
        "VISION_AI_BASE_URL", "https://api.openai.com/v1/chat/completions"
    ).strip()
    model = os.getenv("VISION_AI_MODEL", "gpt-4o-mini").strip()
    enabled = flag not in {"0", "false", "no", "off"}
    return enabled, api_key, base_url, model


def _vision_ai_status() -> dict[str, object]:
    enabled, api_key, base_url, model = _vision_ai_config()
    configured = bool(api_key and base_url)
    with _vision_status_lock:
        runtime = dict(_vision_runtime)
    status = runtime["status"] if enabled and configured else ("disabled" if not enabled else "not_configured")
    return {
        "enabled": enabled and configured,
        "configured": configured,
        "status": status,
        "model": model if enabled and configured else None,
        "last_checked_at": runtime["last_checked_at"],
        "message": runtime["message"] if enabled and configured else (
            "Vision AI dinonaktifkan melalui konfigurasi." if not enabled else "Isi API key dan endpoint Vision AI di file .env."
        ),
    }


def _probe_vision_ai() -> None:
    status, message = "unavailable", "Pemeriksaan endpoint gagal."
    try:
        enabled, api_key, base_url, model = _vision_ai_config()
        if not enabled or not api_key or not base_url:
            status = "disabled" if not enabled else "not_configured"
            message = "Vision AI dinonaktifkan." if not enabled else "API key atau endpoint belum diatur."
        else:
            response = requests.post(
                base_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "max_tokens": 2,
                    "temperature": 0,
                },
                timeout=(3, VISION_AI_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list) or not payload["choices"]:
                raise ValueError("invalid_response")
            status, message = "ready", "Endpoint dan model merespons. Fallback gambar belum diuji dengan foto."
    except requests.Timeout:
        message = "Waktu koneksi habis saat menghubungi endpoint Vision AI."
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 0
        if code in (401, 403):
            message = "Autentikasi ditolak. Periksa API key dan izin model."
        elif code == 404:
            message = "Endpoint atau model tidak ditemukan. Periksa URL dan nama model."
        else:
            message = f"Endpoint menolak permintaan (HTTP {code})."
    except (requests.RequestException, ValueError):
        message = "Tidak dapat terhubung atau respons endpoint tidak valid."
    except Exception:
        message = "Pemeriksaan endpoint gagal karena kesalahan internal."
    with _vision_status_lock:
        _vision_runtime.update(
            status=status,
            last_checked_at=datetime.now(timezone.utc).isoformat(),
            message=message,
        )


def _start_vision_ai_check() -> bool:
    global _vision_check_thread
    with _vision_status_lock:
        if _vision_check_thread and _vision_check_thread.is_alive():
            return False
        _vision_runtime.update(status="checking", message="Sedang menguji endpoint dan model…")
        _vision_check_thread = threading.Thread(target=_probe_vision_ai, name="vision-ai-healthcheck", daemon=True)
        _vision_check_thread.start()
    return True


def _password() -> str:
    value = os.getenv("DASHBOARD_PASSWORD", "")
    if not value:
        raise HTTPException(status_code=503, detail="Set DASHBOARD_PASSWORD in .env")
    return value


def _authenticated(request: Request | WebSocket) -> bool:
    token = request.cookies.get(COOKIE_NAME, "")
    with _state_lock:
        expiry = _session_tokens.get(token, 0) if token else 0
        if expiry > time.time():
            return True
        if token:
            _session_tokens.pop(token, None)
    return False


def require_auth(request: Request) -> None:
    if not _authenticated(request):
        raise HTTPException(status_code=401, detail="Login required")


def require_same_origin(request: Request) -> None:
    # Cookie-authenticated writes must not be triggerable from a different site.
    origin = request.headers.get("origin")
    if not origin or origin != str(request.base_url).rstrip("/"):
        raise HTTPException(status_code=403, detail="Invalid origin")


def _ws_same_origin(websocket: WebSocket) -> bool:
    origin = (websocket.headers.get("origin") or "").rstrip("/")
    host = (websocket.headers.get("host") or "").strip()
    if not origin or not host:
        return False
    return origin in {f"http://{host}", f"https://{host}"}


def _settings() -> CameraSettingsBody:
    _ensure_db()
    with SessionLocal() as db:
        row = db.get(CameraSettings, 1)
        if row is None:
            return CameraSettingsBody(
                video_source=int(os.getenv("VIDEO_SOURCE", "0")) if os.getenv("VIDEO_SOURCE", "0").isdigit() else 0,
                performance_level=int(os.getenv("PERFORMANCE_LEVEL", "1")),
                customer_min_dwell_seconds=0.5,
            )
        return CameraSettingsBody.model_validate(
            {
                "video_source": row.video_source,
                "performance_level": row.performance_level,
                "camera_width": row.camera_width,
                "camera_height": row.camera_height,
                "customer_min_dwell_seconds": row.customer_min_dwell_seconds,
            }
        )


def _start_worker(config: CameraSettingsBody) -> None:
    global _worker, _error, _frame, _generation, _stop, _active_config, _live_counts
    with _frame_ready:
        if _worker is not None and _worker.is_alive():
            raise HTTPException(status_code=409, detail="Kamera sedang berjalan atau berhenti")
        _stop = threading.Event()
        _generation = max(_generation + 1, int(time.time() * 1000))
        _error = None
        _frame = None
        _live_counts = {"staff": 0, "customers": 0, "observing": 0}
        _active_config = config.model_dump()
        _worker = threading.Thread(target=_worker_main, args=(config, _stop), daemon=True, name="stafflens-camera")
        _worker.start()
        _frame_ready.notify_all()
        _notify_ws_waiters()


def _worker_main(config: CameraSettingsBody, stop_event: threading.Event) -> None:
    global _worker, _error, _frame, _fps, _frame_version, _live_counts
    try:
        if stop_event.is_set():
            return
        from app import CameraConfig, main

        def on_frame(image, fps: float, live_counts: dict[str, int] | None = None) -> None:
            global _frame, _fps, _frame_version, _live_counts
            # At most PREVIEW_MAX_FPS preview JPEGs/s; inference and encoding happen once per camera.
            now_mono = time.monotonic()
            if now_mono - getattr(on_frame, "last", 0.0) < 1.0 / PREVIEW_MAX_FPS:
                return
            on_frame.last = now_mono
            if PREVIEW_MAX_WIDTH and image.shape[1] > PREVIEW_MAX_WIDTH:
                scale = PREVIEW_MAX_WIDTH / image.shape[1]
                image = cv2.resize(
                    image, (PREVIEW_MAX_WIDTH, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA
                )
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
            if ok:
                with _frame_ready:
                    had_frame = _frame is not None
                    prev_counts = _live_counts.copy()
                    _frame = encoded.tobytes()
                    _fps = round(fps, 1)
                    if live_counts is not None:
                        _live_counts = {
                            "staff": int(live_counts.get("staff", 0)),
                            "customers": int(live_counts.get("customers", 0)),
                            "observing": int(live_counts.get("observing", 0)),
                        }
                    _frame_version += 1
                    fps_milestone = (
                        abs(_fps - getattr(on_frame, "last_status_fps", -10.0)) >= 0.2
                        and (now_mono - getattr(on_frame, "last_fps_at", 0.0)) >= 2.0
                    )
                    if fps_milestone or not had_frame:
                        on_frame.last_status_fps = _fps
                        on_frame.last_fps_at = now_mono
                    status_changed = (not had_frame) or (_live_counts != prev_counts) or fps_milestone
                    _frame_ready.notify_all()
                    _notify_ws_waiters(frame_only=not status_changed)

        if not stop_event.is_set():
            main(
                stop_event=stop_event,
                on_frame=on_frame,
                on_customer_snapshot=_save_customer_snapshot,
                show_window=False,
                config=CameraConfig(**config.model_dump()),
            )
    except Exception as exc:
        with _state_lock:
            _error = str(exc)
    finally:
        with _frame_ready:
            stop_event.set()
            if _worker is threading.current_thread():
                _worker = None
            _frame = None
            _fps = 0.0
            _live_counts = {"staff": 0, "customers": 0, "observing": 0}
            _frame_ready.notify_all()
            _notify_ws_waiters()


@app.post("/api/auth/login")
def login(body: LoginBody, request: Request) -> Response:
    require_same_origin(request)
    ip = request.client.host if request.client else "local"
    with _state_lock:
        count, until = _login_attempts.get(ip, (0, 0))
        if until > time.time():
            raise HTTPException(status_code=429, detail="Terlalu banyak percobaan. Coba lagi nanti.")
    if not hmac.compare_digest(body.password, _password()):
        with _state_lock:
            count, until = _login_attempts.get(ip, (0, 0))
            if until > time.time():
                raise HTTPException(status_code=429, detail="Terlalu banyak percobaan. Coba lagi nanti.")
            count += 1
            _login_attempts[ip] = (count, time.time() + 300 if count >= 5 else 0)
        raise HTTPException(status_code=401, detail="Password salah")
    token = secrets.token_urlsafe(32)
    with _state_lock:
        _login_attempts.pop(ip, None)
        _session_tokens[token] = time.time() + SESSION_TTL
    response = JSONResponse({"authenticated": True})
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        max_age=SESSION_TTL,
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/auth/session")
def session(request: Request) -> Response:
    return JSONResponse({"authenticated": _authenticated(request)}, headers={"Cache-Control": "no-store"})


@app.post("/api/auth/logout", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def logout(request: Request) -> Response:
    token = request.cookies.get(COOKIE_NAME, "")
    with _state_lock:
        _session_tokens.pop(token, None)
        _notify_ws_waiters()
    response = JSONResponse({"authenticated": False}, headers={"Cache-Control": "no-store"})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


def _query_overview(db: Session) -> dict:
    today_date = datetime.now().astimezone().date()
    start_utc, _ = _today_bounds_utc()

    employees_count = db.scalar(select(func.count(Employee.id)).where(Employee.active.is_(True))) or 0
    embeddings = db.scalar(select(func.count(FaceEmbedding.id))) or 0
    events_count = db.scalar(select(func.count(CameraEvent.id))) or 0
    events_today = (
        db.scalar(select(func.count(CameraEvent.id)).where(CameraEvent.created_at >= start_utc)) or 0
    )

    # Unique daily customers (excluding any row reclassified as an employee)
    customers_today = (
        db.scalar(
            select(func.count(DailyVisitor.id)).where(
                DailyVisitor.visit_date == today_date,
                DailyVisitor.reclassified_employee_id.is_(None),
            )
        )
        or 0
    )
    returning_customers_today = (
        db.scalar(
            select(func.count(DailyVisitor.id)).where(
                DailyVisitor.visit_date == today_date,
                DailyVisitor.reclassified_employee_id.is_(None),
                DailyVisitor.visit_count > 1,
            )
        )
        or 0
    )
    total_customer_visits_today = (
        db.scalar(
            select(func.coalesce(func.sum(DailyVisitor.visit_count), 0)).where(
                DailyVisitor.visit_date == today_date,
                DailyVisitor.reclassified_employee_id.is_(None),
            )
        )
        or 0
    )

    # Distinct active employees detected today
    staff_present_today = (
        db.scalar(
            select(func.count(func.distinct(CameraEvent.employee_id)))
            .join(Employee, Employee.id == CameraEvent.employee_id)
            .where(
                Employee.active.is_(True),
                CameraEvent.created_at >= start_utc,
            )
        )
        or 0
    )

    vision = _vision_ai_status()
    return {
        "customers_today": int(customers_today),
        "returning_customers_today": int(returning_customers_today),
        "total_customer_visits_today": int(total_customer_visits_today),
        "staff_present_today": int(staff_present_today),
        "employees": int(employees_count),
        "embeddings": int(embeddings),
        "events_today": int(events_today),
        "events": int(events_count),
        "camera_id": os.getenv("CAMERA_ID", "local-webcam-01"),
        "today_date": today_date.isoformat(),
        "vision_ai": vision,
    }


def _query_visitors(db: Session, limit: int = 50) -> dict:
    today_date = datetime.now().astimezone().date()
    rows = db.scalars(
        select(DailyVisitor)
        .where(
            DailyVisitor.visit_date == today_date,
            DailyVisitor.reclassified_employee_id.is_(None),
        )
        .order_by(DailyVisitor.last_seen_at.desc(), DailyVisitor.id.desc())
        .limit(max(1, min(limit, 200)))
    ).all()
    items = []
    for row in rows:
        suffix = row.visitor_code.rsplit("-", 1)[-1]
        items.append(
            {
                "id": row.id,
                "code": row.visitor_code,
                "label": f"Customer #{suffix}",
                "visit_count": row.visit_count,
                "first_seen_at": row.first_seen_at.isoformat(),
                "last_seen_at": row.last_seen_at.isoformat(),
                "has_face": row.face_vector is not None,
                "attributes": row.attributes_summary or ("Wajah + Pakaian" if row.face_vector else "Pola Pakaian"),
                "confidence": row.confidence,
                "photo_count": _customer_photo_count(row.visitor_code),
            }
        )
    return {"items": items, "date": today_date.isoformat()}


def _query_events(db: Session, limit: int = 50) -> dict:
    rows = db.scalars(
        select(CameraEvent)
        .order_by(CameraEvent.created_at.desc(), CameraEvent.id.desc())
        .limit(max(1, min(limit, 100)))
    ).all()
    items = []
    for row in rows:
        meta = row.metadata_json or {}
        is_staff = row.employee_id is not None or row.event_type == "identity_locked" or meta.get("kind") == "staff"
        items.append(
            {
                "id": row.id,
                "camera_id": row.camera_id,
                "track_id": row.track_id,
                "employee_id": row.employee_id,
                "label": row.label,
                "confidence": row.confidence,
                "created_at": row.created_at.isoformat(),
                "event_type": row.event_type,
                "kind": "staff" if is_staff else "customer",
                "visit_count": meta.get("visit_count", 1),
                "visitor_code": meta.get("visitor_code"),
            }
        )
    return {"items": items}


def _employee_satusehat_fields(employee: Employee) -> dict:
    return {
        "has_nik": bool(employee.nik_encrypted),
        "nik_masked": f"{'•' * 12}{employee.nik_last4}" if employee.nik_last4 else None,
        "satusehat_practitioner_id": employee.satusehat_practitioner_id,
        "satusehat_synced_at": employee.satusehat_synced_at.isoformat() if employee.satusehat_synced_at else None,
    }


def _query_employees(db: Session) -> dict:
    start_utc, _ = _today_bounds_utc()
    rows = db.execute(
        select(Employee, func.count(FaceEmbedding.id))
        .outerjoin(FaceEmbedding)
        .group_by(Employee.id)
        .order_by(Employee.name)
    ).all()

    today_map: dict[int, tuple[int, datetime | None]] = {}
    presence_rows = db.execute(
        select(
            CameraEvent.employee_id,
            func.count(CameraEvent.id),
            func.max(CameraEvent.created_at),
        )
        .where(CameraEvent.employee_id.is_not(None), CameraEvent.created_at >= start_utc)
        .group_by(CameraEvent.employee_id)
    ).all()
    for emp_id, det_count, last_seen in presence_rows:
        if emp_id is not None:
            today_map[int(emp_id)] = (int(det_count), last_seen)

    items = []
    for employee, emb_count in rows:
        det_today, last_seen_dt = today_map.get(employee.id, (0, None))
        folder = _resolve_employee_folder(employee.employee_code, employee.name)
        photos = _list_folder_photos(folder)
        items.append(
            {
                "id": employee.id,
                "code": employee.employee_code,
                "name": employee.name,
                "role": employee.role,
                "active": employee.active,
                "embeddings": emb_count,
                "folder": f"db_karyawan/{folder.name}",
                "photo_count": len(photos),
                "present_today": det_today > 0,
                "detections_today": det_today,
                "last_seen_today": last_seen_dt.isoformat() if last_seen_dt else None,
                **_employee_satusehat_fields(employee),
            }
        )
    return {"items": items}


@app.get("/api/overview", dependencies=[Depends(require_auth)])
def overview() -> dict:
    _ensure_db()
    with SessionLocal() as db:
        return _query_overview(db)


def _customer_folder(visitor_code: str) -> Path:
    if not re.fullmatch(r"CUST-\d{8}-\d+", visitor_code):
        raise HTTPException(status_code=400, detail="ID customer tidak valid")
    return DB_CUSTOMER_DIR / visitor_code


def _customer_photo_count(visitor_code: str) -> int:
    folder = _customer_folder(visitor_code)
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")


def _save_customer_snapshot(visitor_code: str, crop: np.ndarray, quality: float) -> None:
    """Keep the ten sharpest valid track crops for this daily visitor, local-only."""
    if crop is None or crop.size == 0 or crop.shape[0] < 80 or crop.shape[1] < 60:
        return
    folder = _customer_folder(visitor_code)
    with _CUSTOMER_SNAPSHOT_LOCK:
        folder.mkdir(parents=True, exist_ok=True)
        photos = sorted(folder.glob("*.jpg"))
        if len(photos) >= CUSTOMER_SNAPSHOT_LIMIT:
            try:
                lowest = min(photos, key=lambda p: float(p.stem.split("_", 1)[0]))
            except (ValueError, IndexError):
                lowest = photos[0]
            if quality <= float(lowest.stem.split("_", 1)[0]):
                return
            lowest.unlink(missing_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = folder / f"{quality:.2f}_{stamp}.jpg"
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            output.write_bytes(encoded.tobytes())


@app.get("/api/visitors/{visitor_id}/photos", dependencies=[Depends(require_auth)])
def visitor_photos(visitor_id: int) -> dict:
    _ensure_db()
    with SessionLocal() as db:
        visitor = db.get(DailyVisitor, visitor_id)
        if visitor is None or visitor.reclassified_employee_id is not None:
            raise HTTPException(status_code=404, detail="Customer tidak ditemukan")
        photos = sorted(_customer_folder(visitor.visitor_code).glob("*.jpg"), reverse=True)
        return {"code": visitor.visitor_code, "photos": [
            {"name": p.name, "url": f"/api/visitors/{visitor.id}/photos/{p.name}"} for p in photos
        ]}


@app.get("/api/visitors/{visitor_id}/photos/{filename}", dependencies=[Depends(require_auth)])
def visitor_photo(visitor_id: int, filename: str) -> Response:
    if Path(filename).name != filename or not filename.endswith(".jpg"):
        raise HTTPException(status_code=400, detail="Nama foto tidak valid")
    _ensure_db()
    with SessionLocal() as db:
        visitor = db.get(DailyVisitor, visitor_id)
        if visitor is None or visitor.reclassified_employee_id is not None:
            raise HTTPException(status_code=404, detail="Customer tidak ditemukan")
        folder = _customer_folder(visitor.visitor_code).resolve()
    file_path = (folder / filename).resolve()
    if file_path.parent != folder or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Foto customer tidak ditemukan")
    return FileResponse(file_path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/visitors", dependencies=[Depends(require_auth)])
def visitors(limit: int = 50) -> dict:
    _ensure_db()
    with SessionLocal() as db:
        return _query_visitors(db, limit=limit)


@app.get("/api/events", dependencies=[Depends(require_auth)])
def events(limit: int = 50) -> dict:
    _ensure_db()
    with SessionLocal() as db:
        return _query_events(db, limit=limit)


@app.get("/api/employees", dependencies=[Depends(require_auth)])
def employees() -> dict:
    _ensure_db()
    with SessionLocal() as db:
        return _query_employees(db)


@app.get("/api/employees/{employee_id}", dependencies=[Depends(require_auth)])
def employee_detail(employee_id: int) -> dict:
    _ensure_db()
    with SessionLocal() as db:
        employee = db.get(Employee, employee_id)
        if employee is None:
            raise HTTPException(status_code=404, detail="Pegawai tidak ditemukan")
        emb_count = db.scalar(select(func.count(FaceEmbedding.id)).where(FaceEmbedding.employee_id == employee.id)) or 0
        folder = _resolve_employee_folder(employee.employee_code, employee.name)
        photos = _list_folder_photos(folder)
        return {
            "id": employee.id,
            "code": employee.employee_code,
            "name": employee.name,
            "role": employee.role,
            "active": employee.active,
            "embeddings": int(emb_count),
            "folder": f"db_karyawan/{folder.name}",
            **_employee_satusehat_fields(employee),
            "photos": [
                {
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "url": f"/api/employees/{employee.id}/photos/{p.name}?v={int(p.stat().st_mtime)}",
                }
                for p in photos
            ],
        }


@app.get("/api/employees/{employee_id}/photos/{filename}", dependencies=[Depends(require_auth)])
def employee_photo(employee_id: int, filename: str) -> Response:
    if Path(filename).name != filename or Path(filename).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Nama file foto tidak valid")
    with SessionLocal() as db:
        employee = db.get(Employee, employee_id)
        if employee is None:
            raise HTTPException(status_code=404, detail="Pegawai tidak ditemukan")
        folder = _resolve_employee_folder(employee.employee_code, employee.name)
    file_path = (folder / filename).resolve()
    if not file_path.is_file() or file_path.parent != folder.resolve():
        raise HTTPException(status_code=404, detail="Foto tidak ditemukan di db_karyawan")
    return FileResponse(file_path, headers={"Cache-Control": "no-store"})


def _save_employee_and_photos(employee_id: int | None, body: EmployeeSaveBody) -> dict:
    _ensure_db()
    code = body.code.strip()
    name = body.name.strip()
    role = body.role.strip() or "Karyawan"
    if not code or not name:
        raise HTTPException(status_code=400, detail="Kode dan nama pegawai wajib diisi")

    # NIK: validate and encrypt before touching any file or row.
    nik_update: tuple[str | None, str | None] | None = None  # (encrypted, last4); None = leave unchanged
    if body.nik is not None:
        nik = body.nik.strip()
        if nik == "":
            nik_update = (None, None)
        else:
            if not re.fullmatch(r"\d{16}", nik):
                raise HTTPException(status_code=400, detail="NIK harus 16 digit angka")
            try:
                nik_update = (satusehat.encrypt(nik), nik[-4:])
            except satusehat.SatuSehatError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.message) from None
    with SessionLocal() as db:
        existing_by_code = db.scalar(select(Employee).where(Employee.employee_code == code))
        if employee_id is None:
            if existing_by_code is not None:
                raise HTTPException(status_code=409, detail=f"Kode pegawai {code} sudah terdaftar")
            if not body.new_photos:
                raise HTTPException(status_code=400, detail="Unggah minimal 1 foto wajah untuk pegawai baru")
            folder = DB_KARYAWAN_DIR / _slugify(name)
            if folder.exists() and _list_folder_photos(folder):
                folder = DB_KARYAWAN_DIR / f"{_slugify(name)}_{_slugify(code)}"
            folder.mkdir(parents=True, exist_ok=True)
        else:
            employee = db.get(Employee, employee_id)
            if employee is None:
                raise HTTPException(status_code=404, detail="Pegawai tidak ditemukan")
            if existing_by_code is not None and existing_by_code.id != employee_id:
                raise HTTPException(status_code=409, detail=f"Kode pegawai {code} sudah digunakan pegawai lain")
            old_folder = _resolve_employee_folder(employee.employee_code, employee.name)
            target_folder = DB_KARYAWAN_DIR / _slugify(name)
            if old_folder.exists() and old_folder != target_folder and not target_folder.exists():
                try:
                    old_folder.rename(target_folder)
                    folder = target_folder
                except OSError:
                    folder = old_folder
            else:
                folder = old_folder
            folder.mkdir(parents=True, exist_ok=True)

        # Delete any requested photos from db_karyawan/<folder>
        deleted_count = 0
        for del_name in body.delete_photos:
            if Path(del_name).name != del_name:
                continue
            target_file = (folder / del_name).resolve()
            if target_file.is_file() and target_file.parent == folder.resolve():
                target_file.unlink(missing_ok=True)
                deleted_count += 1

        # Save newly uploaded photos into db_karyawan/<folder>
        existing_files = _list_folder_photos(folder)
        used_numbers = [
            int(p.stem) for p in existing_files if p.stem.isdigit()
        ]
        next_num = (max(used_numbers) if used_numbers else len(existing_files)) + 1

        saved_files: list[Path] = []
        for item in body.new_photos:
            raw_bytes = _decode_data_url(item.data_url)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail=f"File {item.filename} bukan gambar yang valid")
            ext = Path(item.filename).suffix.lower()
            if ext not in ALLOWED_IMAGE_SUFFIXES:
                ext = ".jpg"
            while (folder / f"{next_num}{ext}").exists():
                next_num += 1
            out_path = folder / f"{next_num}{ext}"
            ok, encoded = cv2.imencode(ext if ext != ".jpeg" else ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise HTTPException(status_code=400, detail=f"Gagal menyimpan gambar {item.filename}")
            out_path.write_bytes(encoded.tobytes())
            saved_files.append(out_path)
            next_num += 1

        photos_changed = bool(saved_files or deleted_count or body.reprocess_embeddings or employee_id is None)
        vectors: list[list[float]] = []
        accepted: list[str] = []
        skipped: list[dict[str, str]] = []

        if photos_changed:
            vectors, accepted, skipped = _extract_vectors_from_folder(folder)
            if not vectors:
                # Rollback newly written files if creating a new employee failed face extraction
                for f_path in saved_files:
                    f_path.unlink(missing_ok=True)
                skip_reasons = "; ".join(f"{s['file']}: {s['reason']}" for s in skipped) or "Tidak ada foto valid"
                raise HTTPException(
                    status_code=400,
                    detail=f"Gagal mengekstrak wajah ({skip_reasons}). Gunakan foto jelas dengan tepat 1 wajah.",
                )

        if employee_id is None:
            employee = Employee(employee_code=code, name=name, role=role, active=body.active)
            db.add(employee)
            db.flush()
        else:
            employee = db.get(Employee, employee_id)
            assert employee is not None
            employee.employee_code = code
            employee.name = name
            employee.role = role
            employee.active = body.active

        if nik_update is not None:
            employee.nik_encrypted, employee.nik_last4 = nik_update
            # A changed/cleared NIK invalidates the previous SatuSehat mapping.
            employee.satusehat_practitioner_id = None
            employee.satusehat_synced_at = None

        if photos_changed:
            db.query(FaceEmbedding).filter_by(employee_id=employee.id).delete()
            for vec in vectors:
                db.add(FaceEmbedding(employee_id=employee.id, vector=vec, model_name=_face_model_name()))

        db.commit()
        total_emb = db.scalar(select(func.count(FaceEmbedding.id)).where(FaceEmbedding.employee_id == employee.id)) or 0
        final_photos = _list_folder_photos(folder)

        return {
            "id": employee.id,
            "code": employee.employee_code,
            "name": employee.name,
            "role": employee.role,
            "active": employee.active,
            "embeddings": int(total_emb),
            "folder": f"db_karyawan/{folder.name}",
            **_employee_satusehat_fields(employee),
            "accepted_photos": accepted,
            "skipped_photos": skipped,
            "photos": [
                {
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "url": f"/api/employees/{employee.id}/photos/{p.name}?v={int(p.stat().st_mtime)}",
                }
                for p in final_photos
            ],
        }


@app.post("/api/employees", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def create_employee(body: EmployeeSaveBody) -> dict:
    return _save_employee_and_photos(None, body)


@app.put("/api/employees/{employee_id}", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def update_employee(employee_id: int, body: EmployeeSaveBody) -> dict:
    return _save_employee_and_photos(employee_id, body)


@app.delete("/api/employees/{employee_id}", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def delete_employee(employee_id: int) -> dict:
    _ensure_db()
    with SessionLocal() as db:
        employee = db.get(Employee, employee_id)
        if employee is None:
            raise HTTPException(status_code=404, detail="Pegawai tidak ditemukan")

        folder = _resolve_employee_folder(employee.employee_code, employee.name).resolve()
        root = DB_KARYAWAN_DIR.resolve()
        if folder != root and root in folder.parents and folder.is_dir():
            shutil.rmtree(folder)

        # Keep historical camera events, but detach them from the removed employee.
        db.query(CameraEvent).filter(CameraEvent.employee_id == employee_id).update(
            {CameraEvent.employee_id: None}, synchronize_session=False
        )
        db.delete(employee)
        _audit(db, "employee_deleted", f"{employee.employee_code}: {employee.name}")
        db.commit()
        return {"deleted": True, "employee_id": employee_id}


def _audit(db: Session, action: str, detail: str) -> None:
    db.add(AuditLog(action=action, actor="dashboard", detail=detail[:1000]))


def _satusehat_error(exc: satusehat.SatuSehatError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)


def _mask_secret(row: SatuSehatSettings | None) -> str | None:
    if row is None or not row.client_secret_encrypted:
        return None
    try:
        secret = satusehat.decrypt(row.client_secret_encrypted)
    except satusehat.SatuSehatError:
        return "•" * 12
    return f"{'•' * 12}{secret[-4:]}" if len(secret) >= 12 else "•" * 12


def _satusehat_view(row: SatuSehatSettings | None) -> dict:
    env = row.env if row else "dev"
    return {
        "env": env,
        "client_id": row.client_id if row else "",
        "organization_id": row.organization_id if row else "",
        "client_secret_masked": _mask_secret(row),
        "has_client_secret": bool(row and row.client_secret_encrypted),
        "location_name": row.location_name if row else "",
        "location_id": row.location_id if row else None,
        "location_synced_at": row.location_synced_at.isoformat() if row and row.location_synced_at else None,
        "last_test_at": row.last_test_at.isoformat() if row and row.last_test_at else None,
        "last_test_ok": row.last_test_ok if row else None,
        "last_test_message": row.last_test_message if row else None,
        "base_url": satusehat.base_url(env),
        "encryption_key_configured": satusehat.encryption_ready(),
        "environments": {name: satusehat.base_url(name) for name in satusehat.ENVIRONMENTS},
    }


def _satusehat_config(row: SatuSehatSettings | None) -> satusehat.Config:
    if row is None or not (row.client_id and row.organization_id and row.client_secret_encrypted):
        raise HTTPException(status_code=400, detail="Lengkapi Client ID, Client Secret, dan Organization ID dulu.")
    try:
        secret = satusehat.decrypt(row.client_secret_encrypted)
    except satusehat.SatuSehatError as exc:
        raise _satusehat_error(exc) from None
    return satusehat.Config(
        env=row.env, client_id=row.client_id, client_secret=secret, organization_id=row.organization_id
    )


@app.get("/api/satusehat/settings", dependencies=[Depends(require_auth)])
def satusehat_get_settings() -> dict:
    _ensure_db()
    with SessionLocal() as db:
        return _satusehat_view(db.get(SatuSehatSettings, 1))


@app.put("/api/satusehat/settings", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def satusehat_put_settings(body: SatuSehatSettingsBody) -> dict:
    _ensure_db()
    client_id = body.client_id.strip()
    organization_id = body.organization_id.strip()
    if organization_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", organization_id):
        raise HTTPException(status_code=400, detail="Organization ID hanya boleh huruf, angka, titik, minus, garis bawah")
    new_secret = (body.client_secret or "").strip()
    encrypted: str | None = None
    if new_secret:
        try:
            encrypted = satusehat.encrypt(new_secret)
        except satusehat.SatuSehatError as exc:
            raise _satusehat_error(exc) from None
    with SessionLocal() as db:
        row = db.get(SatuSehatSettings, 1)
        if row is None:
            row = SatuSehatSettings(id=1)
            db.add(row)
        # Any identity/environment change invalidates the saved location and last test result.
        identity_changed = (row.env, row.client_id, row.organization_id) != (body.env, client_id, organization_id)
        row.env = body.env
        row.client_id = client_id
        row.organization_id = organization_id
        if encrypted is not None:
            row.client_secret_encrypted = encrypted
        if identity_changed or encrypted is not None:
            row.last_test_at = None
            row.last_test_ok = None
            row.last_test_message = None
        if identity_changed:
            row.location_id = None
            row.location_synced_at = None
        row.location_name = body.location_name.strip()
        row.updated_at = datetime.now(timezone.utc)
        _audit(db, "satusehat_settings_updated", f"env={body.env} secret_changed={encrypted is not None}")
        db.commit()
        satusehat.clear_token_cache()
        return _satusehat_view(row)


@app.get("/api/satusehat/status", dependencies=[Depends(require_auth)])
def satusehat_status() -> dict:
    _ensure_db()
    with SessionLocal() as db:
        view = _satusehat_view(db.get(SatuSehatSettings, 1))
    return {
        key: view[key]
        for key in (
            "env",
            "base_url",
            "has_client_secret",
            "encryption_key_configured",
            "last_test_at",
            "last_test_ok",
            "last_test_message",
            "location_id",
        )
    }


@app.post("/api/vision-ai/test", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def vision_ai_test() -> dict:
    _start_vision_ai_check()
    return _vision_ai_status()


@app.post("/api/satusehat/test", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def satusehat_test() -> dict:
    _ensure_db()
    with SessionLocal() as db:
        row = db.get(SatuSehatSettings, 1)
        cfg = _satusehat_config(row)
        assert row is not None
        try:
            org_name = satusehat.test_connection(cfg)
            ok, message = True, f"Terhubung. Organisasi: {org_name}"
        except satusehat.SatuSehatError as exc:
            ok, message = False, exc.message
        row.last_test_at = datetime.now(timezone.utc)
        row.last_test_ok = ok
        row.last_test_message = message[:500]
        _audit(db, "satusehat_connection_test", f"env={row.env} ok={ok}")
        db.commit()
        return {"ok": ok, "message": message, "tested_at": row.last_test_at.isoformat()}


@app.post("/api/satusehat/location/sync", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def satusehat_sync_location() -> dict:
    _ensure_db()
    with SessionLocal() as db:
        row = db.get(SatuSehatSettings, 1)
        cfg = _satusehat_config(row)
        assert row is not None
        name = row.location_name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Isi nama lokasi (ruangan/kamera) dulu.")
        code = os.getenv("CAMERA_ID", "local-webcam-01")
        try:
            location_id = satusehat.ensure_location(cfg, code, name)
        except satusehat.SatuSehatError as exc:
            raise _satusehat_error(exc) from None
        row.location_id = location_id
        row.location_synced_at = datetime.now(timezone.utc)
        _audit(db, "satusehat_location_synced", f"env={row.env} code={code}")
        db.commit()
        return {"location_id": location_id, "synced_at": row.location_synced_at.isoformat()}


@app.post(
    "/api/employees/{employee_id}/satusehat/sync",
    dependencies=[Depends(require_auth), Depends(require_same_origin)],
)
def satusehat_sync_employee(employee_id: int) -> dict:
    _ensure_db()
    with SessionLocal() as db:
        cfg = _satusehat_config(db.get(SatuSehatSettings, 1))
        employee = db.get(Employee, employee_id)
        if employee is None:
            raise HTTPException(status_code=404, detail="Pegawai tidak ditemukan")
        if not employee.nik_encrypted:
            raise HTTPException(status_code=400, detail="Isi NIK pegawai dan simpan dulu sebelum sinkronisasi.")
        try:
            nik = satusehat.decrypt(employee.nik_encrypted)
            practitioner_id = satusehat.find_practitioner_id(cfg, nik)
        except satusehat.SatuSehatError as exc:
            raise _satusehat_error(exc) from None
        if practitioner_id is None:
            raise HTTPException(
                status_code=404, detail="NIK ini tidak terdaftar sebagai Practitioner di SatuSehat."
            )
        employee.satusehat_practitioner_id = practitioner_id
        employee.satusehat_synced_at = datetime.now(timezone.utc)
        _audit(db, "satusehat_practitioner_synced", f"employee_id={employee.id} env={cfg.env}")
        db.commit()
        return _employee_satusehat_fields(employee)


@app.get("/api/camera/settings", dependencies=[Depends(require_auth)])
def camera_settings() -> dict:
    return _settings().model_dump()


@app.put("/api/camera/settings", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def update_camera_settings(body: CameraSettingsBody) -> dict:
    global _restarting
    if not _apply_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Pengaturan kamera sedang diterapkan")
    try:
        with _state_lock:
            worker = _worker
            was_running = worker is not None and worker.is_alive() and not _stop.is_set()
            if was_running:
                _restarting = True
            if worker is not None and worker.is_alive():
                _stop.set()
                _frame_ready.notify_all()
                _notify_ws_waiters()
        if worker is not None and worker.is_alive():
            worker.join(timeout=15)
            if worker.is_alive():
                raise HTTPException(status_code=409, detail="Kamera belum berhenti; pengaturan tidak disimpan")
        _ensure_db()
        with SessionLocal.begin() as db:
            row = db.get(CameraSettings, 1)
            if row is None:
                row = CameraSettings(id=1)
                db.add(row)
            for key, value in body.model_dump().items():
                setattr(row, key, value)
        if was_running:
            _start_worker(body)
        return {"settings": body.model_dump(), "running": was_running}
    finally:
        with _state_lock:
            if _restarting:
                _restarting = False
                _notify_ws_waiters()
        _apply_lock.release()


@app.get("/api/system/resources", dependencies=[Depends(require_auth)])
def system_resources() -> dict:
    global _resource_cache
    with _resource_lock:
        if time.monotonic() - _resource_cache[0] < 2:
            return _resource_cache[1]
        import psutil

        cpu_name = _get_cpu_model_name()
        memory = psutil.virtual_memory()
        process = psutil.Process()
        with _state_lock:
            config = (
                _active_config.copy()
                if _active_config and _worker and _worker.is_alive() and not _stop.is_set()
                else None
            )
        result = {
            "cpu": {
                "model": cpu_name,
                "architecture": platform.machine(),
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(),
                "percent": psutil.cpu_percent(interval=None),
            },
            "memory": {
                "total_bytes": memory.total,
                "used_bytes": memory.used,
                "percent": memory.percent,
                "process_bytes": process.memory_info().rss,
            },
            "camera": {
                "source": config["video_source"] if config else None,
                "requested_width": config["camera_width"] if config else None,
                "requested_height": config["camera_height"] if config else None,
            },
        }
        _resource_cache = (time.monotonic(), result)
        return result


def _camera_status_dict() -> dict:
    """Build camera status snapshot. Caller must hold _state_lock / _frame_ready."""
    running = _worker is not None and _worker.is_alive()
    active = running and not _stop.is_set()
    effective_running = active or _restarting
    return {
        "running": effective_running,
        "stopping": running and _stop.is_set() and not _restarting,
        "fps": _fps if active else 0.0,
        "has_frame": _frame is not None and active,
        "generation": _generation if effective_running else 0,
        "frame_version": _frame_version if active else 0,
        "error": _error,
        "active_config": _active_config if effective_running else None,
        "live_staff": _live_counts["staff"] if active else 0,
        "live_customers": _live_counts["customers"] if active else 0,
        "live_observing": _live_counts["observing"] if active else 0,
    }


def _core_status_signature(status: dict) -> tuple:
    return (
        status["running"],
        status["stopping"],
        status["has_frame"],
        status["generation"],
        status["error"],
        status["live_staff"],
        status["live_customers"],
        status["live_observing"],
    )


@app.get("/api/dashboard", dependencies=[Depends(require_auth)])
def dashboard_bundle(visitor_limit: int = 100, event_limit: int = 60) -> dict:
    _ensure_db()
    with SessionLocal() as db:
        overview_data = _query_overview(db)
        visitors_data = _query_visitors(db, limit=visitor_limit)["items"]
        events_data = _query_events(db, limit=event_limit)["items"]
        employees_data = _query_employees(db)["items"]
    with _state_lock:
        cam_status = _camera_status_dict()
    return {
        "overview": overview_data,
        "visitors": visitors_data,
        "events": events_data,
        "employees": employees_data,
        "camera": cam_status,
    }


@app.get("/api/camera/status", dependencies=[Depends(require_auth)])
def camera_status() -> dict:
    with _state_lock:
        return _camera_status_dict()


@app.post("/api/camera/start", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def camera_start() -> dict:
    _password()
    if not _apply_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Pengaturan kamera sedang diterapkan")
    try:
        with _state_lock:
            if _worker is not None and _worker.is_alive():
                if _stop.is_set():
                    raise HTTPException(status_code=409, detail="Kamera sedang berhenti")
                return {"running": True}
        _start_worker(_settings())
        return {"running": True}
    finally:
        _apply_lock.release()


@app.post("/api/camera/stop", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def camera_stop() -> dict:
    global _worker, _frame, _fps
    if not _apply_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Pengaturan kamera sedang diterapkan")
    try:
        with _frame_ready:
            worker = _worker
            _stop.set()
            _frame_ready.notify_all()
            _notify_ws_waiters()
        if worker is not None and worker.is_alive():
            worker.join(timeout=15)
            if worker.is_alive():
                raise HTTPException(status_code=409, detail="Kamera belum berhenti; coba sesaat lagi")
        with _frame_ready:
            if _worker is worker:
                _worker = None
            _frame = None
            _fps = 0.0
            _frame_ready.notify_all()
            _notify_ws_waiters()
        return {"stopping": False, "running": False}
    finally:
        _apply_lock.release()


@app.get("/api/camera/frame", dependencies=[Depends(require_auth)])
def camera_frame() -> Response:
    with _state_lock:
        frame = _frame
    if frame is None:
        raise HTTPException(status_code=404, detail="Belum ada frame kamera")
    return Response(content=frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.websocket("/api/camera/ws")
async def camera_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    if not _ws_same_origin(websocket):
        await websocket.close(code=1008, reason="Invalid origin")
        return
    if not _authenticated(websocket):
        await websocket.close(code=4401, reason="Login required")
        return

    initial_sub = websocket.query_params.get("subscribe", "1") != "0"
    token = websocket.cookies.get(COOKIE_NAME, "")
    loop = asyncio.get_running_loop()
    frame_event = asyncio.Event()
    client_state: dict[str, object] = {
        "subscribe_frames": initial_sub,
        "force_frame": initial_sub,
        "force_status": True,
        "disconnected": False,
    }

    async def _receive_control() -> None:
        try:
            while not client_state["disconnected"]:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if isinstance(data, dict):
                    with _state_lock:
                        changed = False
                        if "subscribe_frames" in data:
                            want = bool(data["subscribe_frames"])
                            prev_want = bool(client_state["subscribe_frames"])
                            client_state["subscribe_frames"] = want
                            if want and not prev_want:
                                client_state["force_frame"] = True
                                changed = True
                            elif want != prev_want:
                                changed = True
                        if data.get("force_frame") and client_state["subscribe_frames"]:
                            client_state["force_frame"] = True
                            changed = True
                        if data.get("force_status"):
                            client_state["force_status"] = True
                            changed = True
                        if changed:
                            frame_event.set()
        except Exception:
            pass
        finally:
            with _state_lock:
                client_state["disconnected"] = True
                frame_event.set()

    with _state_lock:
        _ws_waiters[frame_event] = (loop, client_state)
        frame_event.set()

    recv_task = asyncio.create_task(_receive_control())
    last_version = -1
    last_core_sig: tuple | None = None
    last_sent_fps = -1.0
    last_fps_sent_at = 0.0
    try:
        while not client_state["disconnected"]:
            now_mono = time.monotonic()
            now_wall = time.time()
            with _state_lock:
                frame_event.clear()
                expiry = _session_tokens.get(token, 0.0) if token else 0.0
                session_valid = expiry > now_wall
                if not session_valid and token:
                    _session_tokens.pop(token, None)
                status = _camera_status_dict()
                core_sig = _core_status_signature(status)
                want_frames = bool(client_state["subscribe_frames"])
                force_frame = bool(client_state["force_frame"])
                force_status = bool(client_state["force_status"])

                has_new_frame = (
                    want_frames
                    and status["has_frame"]
                    and _frame is not None
                    and (_frame_version != last_version or force_frame)
                )
                frame_bytes = _frame if has_new_frame else None
                if frame_bytes is not None:
                    last_version = _frame_version
                    client_state["force_frame"] = False

                fps_interval = 1.0 if want_frames else 2.0
                core_changed = (core_sig != last_core_sig) or force_status
                fps_changed = (
                    status["running"]
                    and abs(status["fps"] - last_sent_fps) >= 0.2
                    and (now_mono - last_fps_sent_at) >= fps_interval
                )
                if core_changed or fps_changed:
                    status_payload = status
                    last_core_sig = core_sig
                    last_sent_fps = status["fps"]
                    last_fps_sent_at = now_mono
                    client_state["force_status"] = False
                else:
                    status_payload = None

            if not session_valid:
                await websocket.close(code=4401, reason="Session expired")
                break

            if status_payload is not None and not client_state["disconnected"]:
                await websocket.send_json({"type": "status", **status_payload})

            if frame_bytes is not None and client_state["subscribe_frames"] and not client_state["disconnected"]:
                await websocket.send_bytes(frame_bytes)

            if client_state["disconnected"]:
                break

            wait_timeout = max(0.5, expiry - time.time())
            try:
                await asyncio.wait_for(frame_event.wait(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass
    finally:
        with _state_lock:
            client_state["disconnected"] = True
            _ws_waiters.pop(frame_event, None)
        recv_task.cancel()
        try:
            await recv_task
        except (asyncio.CancelledError, Exception):
            pass


@app.get("/api/camera/stream", dependencies=[Depends(require_auth)])
async def camera_stream(request: Request) -> Response:
    token = request.cookies.get(COOKIE_NAME, "")
    with _state_lock:
        generation = _generation
        worker = _worker
        if worker is None or not worker.is_alive() or _stop.is_set():
            raise HTTPException(status_code=404, detail="Kamera belum berjalan")

    loop = asyncio.get_running_loop()
    frame_event = asyncio.Event()
    stream_state: dict[str, object] = {"subscribe_frames": True}

    async def frames():
        last_version = -1
        with _state_lock:
            _ws_waiters[frame_event] = (loop, stream_state)
            frame_event.set()
        try:
            while True:
                if await request.is_disconnected():
                    return
                with _state_lock:
                    frame_event.clear()
                    if _stop.is_set() or _generation != generation or not worker.is_alive():
                        return
                    if _session_tokens.get(token, 0) <= time.time():
                        _session_tokens.pop(token, None)
                        return
                    if _frame is not None and _frame_version != last_version:
                        frame = _frame
                        last_version = _frame_version
                    else:
                        frame = None
                if frame is not None:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                try:
                    await asyncio.wait_for(frame_event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
        finally:
            with _state_lock:
                _ws_waiters.pop(frame_event, None)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "X-Content-Type-Options": "nosniff"},
    )


if (DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str) -> Response:
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("DASHBOARD_PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("DASHBOARD_PORT must be between 1 and 65535")
    uvicorn.run(app, host="127.0.0.1", port=port, workers=1, timeout_keep_alive=65)
