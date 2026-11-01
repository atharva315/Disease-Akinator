"""Neon + MongoDB integration test for doctor registration.

Run:
    python tests/test_doctor_registration_integration.py

Uses a unique temporary account and synthetic PDF content. It verifies the
PostgreSQL doctor row, Argon2id password hash, MongoDB GridFS PDF, duplicate
email rejection, invalid PDF rejection, and cleanup in both databases.
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from psycopg.conninfo import make_conninfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("=== Disease Akinator Doctor Registration Integration Test ===")

    pg_uri = getpass.getpass("Paste Neon application URI (hidden): ").strip()
    parsed_pg = urlsplit(pg_uri)
    if unquote(parsed_pg.username or "") != "disease_akinator_app":
        print("FAIL: Neon URI must use disease_akinator_app.")
        return 1
    if parsed_pg.path.lstrip("/").split("?")[0] != "disease_akinator":
        print("FAIL: Neon URI must target disease_akinator.")
        return 1

    pg_password = getpass.getpass("Neon application password (hidden): ")
    if not pg_password:
        print("FAIL: Neon password cannot be empty.")
        return 1
    os.environ["DISEASE_AKINATOR_POSTGRES_URL"] = make_conninfo(
        pg_uri, password=pg_password
    )

    mongo_uri = getpass.getpass("Paste MongoDB Atlas application URI (hidden): ").strip()
    parsed_mongo = urlsplit(mongo_uri)
    if parsed_mongo.scheme not in {"mongodb", "mongodb+srv"}:
        print("FAIL: Invalid MongoDB URI scheme.")
        return 1
    if unquote(parsed_mongo.username or "") != "disease_akinator_app":
        print("FAIL: MongoDB URI must use disease_akinator_app.")
        return 1
    os.environ["DISEASE_AKINATOR_MONGODB_URI"] = mongo_uri

    from database.mongodb import (
        DEGREE_BUCKET,
        delete_file,
        get_file_metadata,
    )
    from database.postgres import (
        delete_doctor,
        get_doctor_auth_record,
        get_doctor_by_email,
    )
    from services.auth_service import authenticate_doctor
    from services.doctor_service import (
        DuplicateDoctorEmailError,
        register_doctor,
    )

    run_id = uuid4().hex
    email = f"phase5-register-{run_id}@example.invalid"
    password = f"Pa9!{uuid4().hex}Z"
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"Temporary synthetic doctor-registration integration-test document.\n"
        b"%%EOF\n"
    )
    doctor_id = None
    degree_file_id = None

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        print(f"PASS: {label}")

    try:
        doctor = register_doctor(
            full_name="Temporary Phase 5 Doctor",
            age=35,
            degree="MBBS",
            custom_degree=None,
            hospital="Temporary Phase 5 Clinic",
            city="Pune",
            email=email.upper(),
            password=password,
            degree_filename="phase5-degree.pdf",
            degree_file_bytes=pdf_bytes,
        )
        doctor_id = doctor["id"]
        degree_file_id = doctor["degree_file_id"]
        check(
            doctor["email"] == email,
            "Register doctor and normalize email",
        )
        check(
            doctor["degree_file_id"] == degree_file_id
            and len(degree_file_id) == 24,
            "Store MongoDB GridFS degree file reference in PostgreSQL",
        )

        stored = get_doctor_by_email(email)
        check(
            stored is not None
            and stored["id"] == doctor_id
            and stored["degree_filename"] == "phase5-degree.pdf",
            "Read the newly registered doctor profile from Neon",
        )

        auth_record = get_doctor_auth_record(email)
        check(
            auth_record is not None
            and auth_record["password_hash"].startswith("$argon2id$")
            and auth_record["password_hash"] != password,
            "Verify Neon stores an Argon2id hash, not the plaintext password",
        )

        metadata = get_file_metadata(DEGREE_BUCKET, degree_file_id)
        check(
            metadata is not None
            and metadata["metadata"]["uploaded_by_email"] == email
            and metadata["metadata"]["file_type"] == "doctor_degree",
            "Verify the PDF exists in GridFS with uploader metadata",
        )

        profile = authenticate_doctor(email, password)
        check(
            profile is not None
            and profile["id"] == doctor_id
            and "password_hash" not in profile,
            "Newly registered doctor can log in and receives a safe profile",
        )

        duplicate_rejected = False
        try:
            register_doctor(
                full_name="Duplicate Temporary Doctor",
                age=35,
                degree="MBBS",
                custom_degree=None,
                hospital=None,
                city=None,
                email=email,
                password=password,
                degree_filename="duplicate-degree.pdf",
                degree_file_bytes=pdf_bytes,
            )
        except DuplicateDoctorEmailError:
            duplicate_rejected = True
        check(
            duplicate_rejected,
            "Reject duplicate email without creating a second account",
        )

        invalid_pdf_rejected = False
        try:
            register_doctor(
                full_name="Invalid PDF Test Doctor",
                age=35,
                degree="MBBS",
                custom_degree=None,
                hospital=None,
                city=None,
                email=f"invalid-{run_id}@example.invalid",
                password=password,
                degree_filename="not-a-pdf.pdf",
                degree_file_bytes=b"This is not a PDF",
            )
        except ValueError:
            invalid_pdf_rejected = True
        check(
            invalid_pdf_rejected,
            "Reject invalid PDF content before upload",
        )

        print("\n--- Cleanup ---")
        check(delete_doctor(doctor_id), "Delete temporary doctor from Neon")
        doctor_id = None
        check(
            delete_file(DEGREE_BUCKET, degree_file_id),
            "Delete temporary degree PDF from MongoDB GridFS",
        )
        degree_file_id = None
        print("\n==============================================")
        print("ALL DOCTOR REGISTRATION INTEGRATION TESTS PASSED")
        print("==============================================")
        return 0

    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        return 1

    finally:
        if doctor_id is not None:
            try:
                delete_doctor(doctor_id)
                print("Cleanup: temporary doctor removed from Neon.")
            except Exception as exc:
                print(f"WARNING: Neon cleanup may be required: {type(exc).__name__}")

        if degree_file_id is not None:
            try:
                delete_file(DEGREE_BUCKET, degree_file_id)
                print("Cleanup: temporary degree PDF removed from GridFS.")
            except Exception as exc:
                print(f"WARNING: MongoDB cleanup may be required: {type(exc).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
