"""Register employee face photos into the StaffLens database."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

from database import Employee, FaceEmbedding, SessionLocal, initialize_database

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Register employee face photos.")
    parser.add_argument("--code", required=True, help="Unique employee code, e.g. EMP-001")
    parser.add_argument("--name", required=True, help="Employee display name")
    parser.add_argument("--role", default="Karyawan", help="Role label")
    parser.add_argument("--photos", required=True, help="Folder containing employee photos")
    args = parser.parse_args()

    photos = sorted(path for path in Path(args.photos).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not photos:
        raise SystemExit("No .jpg/.jpeg/.png photos found.")

    face_model = os.getenv("FACE_MODEL", "buffalo_l").strip() or "buffalo_l"
    face_app = FaceAnalysis(
        name=face_model, providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    vectors: list[list[float]] = []

    for photo in photos:
        image = cv2.imread(str(photo))
        faces = face_app.get(image) if image is not None else []
        if len(faces) != 1:
            print(f"Skipped {photo.name}: expected exactly 1 face, got {len(faces)}.")
            continue
        vectors.append(np.asarray(faces[0].embedding, dtype=np.float32).tolist())
        print(f"Accepted {photo.name}")

    if not vectors:
        raise SystemExit("No usable photo. Add clear single-face photos and retry.")

    initialize_database()
    with SessionLocal() as session:
        employee = session.query(Employee).filter_by(employee_code=args.code).one_or_none()
        if employee is None:
            employee = Employee(employee_code=args.code, name=args.name, role=args.role)
            session.add(employee)
            session.flush()
        else:
            employee.name, employee.role, employee.active = args.name, args.role, True
            session.query(FaceEmbedding).filter_by(employee_id=employee.id, model_name=face_model).delete()

        for vector in vectors:
            session.add(FaceEmbedding(employee_id=employee.id, vector=vector, model_name=face_model))
        session.commit()

    print(f"Saved {len(vectors)} face embedding(s) for {args.name}.")


if __name__ == "__main__":
    main()
