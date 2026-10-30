"""Database schema and persistence helpers for StaffLens AI."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
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


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    initialize_database()
    print("Database schema ready.")
