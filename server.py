"""Password-protected local dashboard API, employee photo registration, and one-camera worker."""

from __future__ import annotations

import asyncio
import base64
import hmac
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

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from database import (
    CameraEvent,
    CameraSettings,
    DailyVisitor,
    Employee,
    FaceEmbedding,
    SessionLocal,
    initialize_database,
)

load_dotenv()

COOKIE_NAME = "stafflens_session"
SESSION_TTL = 8 * 60 * 60
ROOT_DIR = Path(__file__).parent
DIST = ROOT_DIR / "frontend" / "dist"
DB_KARYAWAN_DIR = ROOT_DIR / "db_karyawan"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

app = FastAPI(title="StaffLens AI", docs_url=None, redoc_url=None, openapi_url=None)


class _SuppressFramePollFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/camera/frame" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_SuppressFramePollFilter())


@app.on_event("startup")
def _install_log_filter() -> None:
    logging.getLogger("uvicorn.access").addFilter(_SuppressFramePollFilter())


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

        analyzer = FaceAnalysis(name=os.getenv("FACE_MODEL", "buffalo_l"), providers=["CPUExecutionProvider"])
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


class PhotoUploadItem(BaseModel):
    filename: str
    data_url: str


class EmployeeSaveBody(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    role: str = Field(default="Karyawan", min_length=1, max_length=100)
    active: bool = True
    new_photos: list[PhotoUploadItem] = Field(default_factory=list)
    delete_photos: list[str] = Field(default_factory=list)
    reprocess_embeddings: bool = False


def _today_bounds_utc() -> tuple[datetime, datetime]:
    now_local = datetime.now().astimezone()
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_local.astimezone(timezone.utc), now_local.astimezone(timezone.utc)


def _vision_ai_status() -> dict[str, object]:
    flag = os.getenv("VISION_AI_ENABLED", "auto").strip().lower()
    api_key = os.getenv("VISION_AI_API_KEY", "").strip()
    base_url = os.getenv("VISION_AI_API_KEY", "") and os.getenv(
        "VISION_AI_BASE_URL", "https://api.openai.com/v1/chat/completions"
    ).strip()
    model = os.getenv("VISION_AI_MODEL", "gpt-4o-mini").strip()
    enabled = flag not in {"0", "false", "no", "off"} and bool(api_key and base_url)
    return {"enabled": enabled, "model": model if enabled else None, "configured": bool(api_key)}


def _password() -> str:
    value = os.getenv("DASHBOARD_PASSWORD", "")
    if not value:
        raise HTTPException(status_code=503, detail="Set DASHBOARD_PASSWORD in .env")
    return value


def _authenticated(request: Request) -> bool:
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


def _settings() -> CameraSettingsBody:
    initialize_database()
    with SessionLocal() as db:
        row = db.get(CameraSettings, 1)
        if row is None:
            return CameraSettingsBody(
                video_source=int(os.getenv("VIDEO_SOURCE", "0")) if os.getenv("VIDEO_SOURCE", "0").isdigit() else 0,
                performance_level=int(os.getenv("PERFORMANCE_LEVEL", "1")),
            )
        return CameraSettingsBody.model_validate(
            {
                "video_source": row.video_source,
                "performance_level": row.performance_level,
                "camera_width": row.camera_width,
                "camera_height": row.camera_height,
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


def _worker_main(config: CameraSettingsBody, stop_event: threading.Event) -> None:
    global _error, _frame, _fps, _frame_version, _live_counts
    try:
        if stop_event.is_set():
            return
        from app import CameraConfig, main

        def on_frame(image, fps: float, live_counts: dict[str, int] | None = None) -> None:
            global _frame, _fps, _frame_version, _live_counts
            # At most 8 preview JPEGs/s; inference and encoding happen once per camera.
            if time.monotonic() - getattr(on_frame, "last", 0) < 0.125:
                return
            on_frame.last = time.monotonic()
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ok:
                with _frame_ready:
                    _frame = encoded.tobytes()
                    _fps = round(fps, 1)
                    if live_counts is not None:
                        _live_counts = {
                            "staff": int(live_counts.get("staff", 0)),
                            "customers": int(live_counts.get("customers", 0)),
                            "observing": int(live_counts.get("observing", 0)),
                        }
                    _frame_version += 1
                    _frame_ready.notify_all()

        if not stop_event.is_set():
            main(stop_event=stop_event, on_frame=on_frame, show_window=False, config=CameraConfig(**config.model_dump()))
    except Exception as exc:
        with _state_lock:
            _error = str(exc)
    finally:
        with _frame_ready:
            _frame = None
            _fps = 0.0
            _live_counts = {"staff": 0, "customers": 0, "observing": 0}
            _frame_ready.notify_all()


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
    response = JSONResponse({"authenticated": False}, headers={"Cache-Control": "no-store"})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@app.get("/api/overview", dependencies=[Depends(require_auth)])
def overview() -> dict:
    initialize_database()
    today_date = datetime.now().astimezone().date()
    start_utc, _ = _today_bounds_utc()

    with SessionLocal() as db:
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


@app.get("/api/visitors", dependencies=[Depends(require_auth)])
def visitors(limit: int = 50) -> dict:
    initialize_database()
    today_date = datetime.now().astimezone().date()
    with SessionLocal() as db:
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
                }
            )
        return {"items": items, "date": today_date.isoformat()}


@app.get("/api/events", dependencies=[Depends(require_auth)])
def events(limit: int = 50) -> dict:
    initialize_database()
    with SessionLocal() as db:
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


@app.get("/api/employees", dependencies=[Depends(require_auth)])
def employees() -> dict:
    initialize_database()
    start_utc, _ = _today_bounds_utc()
    with SessionLocal() as db:
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
                }
            )
        return {"items": items}


@app.get("/api/employees/{employee_id}", dependencies=[Depends(require_auth)])
def employee_detail(employee_id: int) -> dict:
    initialize_database()
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
    initialize_database()
    code = body.code.strip()
    name = body.name.strip()
    role = body.role.strip() or "Karyawan"
    if not code or not name:
        raise HTTPException(status_code=400, detail="Kode dan nama pegawai wajib diisi")

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

        if photos_changed:
            db.query(FaceEmbedding).filter_by(employee_id=employee.id).delete()
            for vec in vectors:
                db.add(FaceEmbedding(employee_id=employee.id, vector=vec, model_name="buffalo_l"))

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


@app.get("/api/camera/settings", dependencies=[Depends(require_auth)])
def camera_settings() -> dict:
    return _settings().model_dump()


@app.put("/api/camera/settings", dependencies=[Depends(require_auth), Depends(require_same_origin)])
def update_camera_settings(body: CameraSettingsBody) -> dict:
    if not _apply_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Pengaturan kamera sedang diterapkan")
    try:
        with _state_lock:
            worker = _worker
            was_running = worker is not None and worker.is_alive() and not _stop.is_set()
            if worker is not None and worker.is_alive():
                _stop.set()
                _frame_ready.notify_all()
        if worker is not None and worker.is_alive():
            worker.join(timeout=15)
            if worker.is_alive():
                raise HTTPException(status_code=409, detail="Kamera belum berhenti; pengaturan tidak disimpan")
        initialize_database()
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


@app.get("/api/camera/status", dependencies=[Depends(require_auth)])
def camera_status() -> dict:
    with _state_lock:
        running = _worker is not None and _worker.is_alive()
        active = running and not _stop.is_set()
        return {
            "running": active,
            "stopping": running and _stop.is_set(),
            "fps": _fps if running else 0,
            "has_frame": _frame is not None and active,
            "generation": _generation if active else 0,
            "frame_version": _frame_version if active else 0,
            "error": _error,
            "active_config": _active_config if active else None,
            "live_staff": _live_counts["staff"] if active else 0,
            "live_customers": _live_counts["customers"] if active else 0,
            "live_observing": _live_counts["observing"] if active else 0,
        }


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
    if not _apply_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Pengaturan kamera sedang diterapkan")
    try:
        with _frame_ready:
            _stop.set()
            _frame_ready.notify_all()
        return {"stopping": True}
    finally:
        _apply_lock.release()


@app.get("/api/camera/frame", dependencies=[Depends(require_auth)])
def camera_frame() -> Response:
    with _state_lock:
        frame = _frame
    if frame is None:
        raise HTTPException(status_code=404, detail="Belum ada frame kamera")
    return Response(content=frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/camera/stream", dependencies=[Depends(require_auth)])
async def camera_stream(request: Request) -> Response:
    token = request.cookies.get(COOKIE_NAME, "")
    with _state_lock:
        generation = _generation
        worker = _worker
        if worker is None or not worker.is_alive() or _stop.is_set():
            raise HTTPException(status_code=404, detail="Kamera belum berjalan")

    async def frames():
        last_version = -1
        while True:
            if await request.is_disconnected():
                return
            with _state_lock:
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
            await asyncio.sleep(0.08)

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
    uvicorn.run(app, host="127.0.0.1", port=port, workers=1)
