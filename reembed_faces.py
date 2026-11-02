"""Re-generate employee face embeddings for one InsightFace model from photos in db_karyawan/.

Only rows with model_name == --model are replaced; embeddings of other models are kept,
so switching FACE_MODEL back (e.g. buffalo_s -> buffalo_l) needs no re-registration.
See docs/OPTIMASI_PERFORMA.md.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Re-embed employee faces for a given InsightFace model.")
    parser.add_argument("--model", default=None, help="InsightFace model pack (default: FACE_MODEL from .env)")
    args = parser.parse_args()
    model = (args.model or os.getenv("FACE_MODEL", "buffalo_l")).strip()
    os.environ["FACE_MODEL"] = model  # server helpers read FACE_MODEL at call time

    from database import Employee, FaceEmbedding, SessionLocal, initialize_database
    from server import _extract_vectors_from_folder, _resolve_employee_folder

    initialize_database()
    done, missing, failed = [], [], []
    with SessionLocal() as session:
        for employee in session.query(Employee).order_by(Employee.id).all():
            folder = _resolve_employee_folder(employee.employee_code, employee.name)
            vectors, accepted, skipped = _extract_vectors_from_folder(folder)
            label = f"{employee.employee_code} {employee.name}"
            if not accepted and not skipped:
                missing.append(label)
                print(f"[SKIP] {label}: tidak ada foto di {folder}")
                continue
            if not vectors:
                failed.append(label)
                print(f"[FAIL] {label}: tidak ada foto valid ({skipped})")
                continue
            session.query(FaceEmbedding).filter_by(employee_id=employee.id, model_name=model).delete()
            for vec in vectors:
                session.add(FaceEmbedding(employee_id=employee.id, vector=vec, model_name=model))
            session.commit()
            done.append(label)
            print(f"[OK]   {label}: {len(vectors)} embedding ({model})")

    print(f"\nSelesai model={model}: {len(done)} OK, {len(missing)} tanpa foto, {len(failed)} gagal.")
    if missing or failed:
        print("Daftarkan ulang foto karyawan di atas lewat dashboard atau register_employee.py.")


if __name__ == "__main__":
    main()
