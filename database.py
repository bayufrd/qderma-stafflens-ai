"""Database schema and persistence helpers for StaffLens AI."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    role: Mapped[str] = mapped_column(String(100), default="Karyawan")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    # SatuSehat: NIK is stored encrypted (reversible, needed for Practitioner lookup); only last 4 digits in clear.
    nik_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    nik_last4: Mapped[str | None] = mapped_column(String(4), nullable=True)
    satusehat_practitioner_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    satusehat_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    embeddings: Mapped[list["FaceEmbedding"]] = relationship(back_populates="employee", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(100), default="buffalo_l")
    vector: Mapped[list[float]] = mapped_column(JSONB)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    employee: Mapped[Employee] = relationship(back_populates="embeddings")


class CameraEvent(Base):
    __tablename__ = "camera_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(100), index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    employee_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"), nullable=True, index=True)
    label: Mapped[str] = mapped_column(String(255))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), default="identity_locked")
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class DailyVisitor(Base):
    """Unique daily visitor registry (1 row = 1 unique customer per calendar day)."""

    __tablename__ = "daily_visitors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    visit_date: Mapped[date] = mapped_column(Date, index=True)
    visitor_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    face_vector: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
    body_vector: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
    attributes_summary: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    visit_count: Mapped[int] = mapped_column(Integer, default=1)
    last_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reclassified_employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employees.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    review_status: Mapped[str] = mapped_column(String(32), default="active", index=True)


class VisitorFeedback(Base):
    __tablename__ = "visitor_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    visitor_id: Mapped[int] = mapped_column(ForeignKey("daily_visitors.id", ondelete="CASCADE"), index=True)
    reason_code: Mapped[str] = mapped_column(String(64))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    predicted_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), default="dashboard")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    undone_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    action: Mapped[str] = mapped_column(String(100), index=True)
    actor: Mapped[str] = mapped_column(String(255), default="system")
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class CameraSettings(Base):
    __tablename__ = "camera_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    video_source: Mapped[int] = mapped_column(Integer, default=0)
    performance_level: Mapped[int] = mapped_column(Integer, default=5)
    camera_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    camera_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    customer_min_dwell_seconds: Mapped[float] = mapped_column(Float, default=0.5)



class SatuSehatSettings(Base):
    """Single-row SatuSehat integration settings. client_secret is stored encrypted (Fernet)."""

    __tablename__ = "satusehat_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    env: Mapped[str] = mapped_column(String(16), default="dev")
    client_id: Mapped[str] = mapped_column(String(255), default="")
    organization_id: Mapped[str] = mapped_column(String(64), default="")
    client_secret_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    location_name: Mapped[str] = mapped_column(String(255), default="")
    location_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    location_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_test_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_test_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_test_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# Columns added after the first release; create_all() does not alter existing tables.
_DAILY_VISITOR_NEW_COLUMNS = {
    "review_status": "VARCHAR(32) DEFAULT 'active'",
}

_EMPLOYEE_NEW_COLUMNS = {
    "nik_encrypted": "TEXT",
    "nik_last4": "VARCHAR(4)",
    "satusehat_practitioner_id": "VARCHAR(64)",
    "satusehat_synced_at": "TIMESTAMP WITH TIME ZONE",
}


def _migrate_columns(table: str, definitions: dict[str, str]) -> None:
    existing = {col["name"] for col in inspect(engine).get_columns(table)}
    missing = {name: ddl for name, ddl in definitions.items() if name not in existing}
    if not missing:
        return
    is_mysql = engine.dialect.name == "mysql"
    with engine.begin() as conn:
        for name, ddl in missing.items():
            if is_mysql and ddl.startswith("TIMESTAMP"):
                ddl = "DATETIME"
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def _migrate_employee_columns() -> None:
    _migrate_columns("employees", _EMPLOYEE_NEW_COLUMNS)


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
    # Lightweight migration for installations created before dwell-time setting existed.
    with engine.begin() as connection:
        columns = {column["name"] for column in inspect(engine).get_columns("camera_settings")}
        if "customer_min_dwell_seconds" not in columns:
            connection.execute(text("ALTER TABLE camera_settings ADD COLUMN customer_min_dwell_seconds FLOAT DEFAULT 0.5"))
    _migrate_employee_columns()
    _migrate_columns("daily_visitors", _DAILY_VISITOR_NEW_COLUMNS)


if __name__ == "__main__":
    initialize_database()
    print("Database schema ready.")
