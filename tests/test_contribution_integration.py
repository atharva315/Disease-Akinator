"""Neon + MongoDB integration test for doctor contributions.

Run:
    python tests/test_contribution_integration.py

Creates one temporary doctor, submits a symptom contribution and two image
contributions, verifies uploader attribution and exact GridFS download bytes,
checks statistics, then removes all temporary PostgreSQL and GridFS records.
"""

from __future__ import annotations

import base64
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


def make_test_png() -> bytes:
    """Return a valid 2x2 RGB PNG fixture (stdlib-only)."""
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91Jpz"
        "AAAAFklEQVR4nGPcp+HGwMDAxMDAwMDAAAAOsgEwOQnPWQAAAABJRU5ErkJggg==",
        validate=True,
    )
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Embedded PNG test fixture is invalid.")
    return data


def make_test_jpeg() -> bytes:
    """Return a valid 2x2 RGB JPEG fixture (stdlib-only)."""
    data = base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
        "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/"
        "wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/"
        "8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW"
        "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3"
        "+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/"
        "8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RV"
        "VldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3"
        "+Pn6/9oADAMBAAIRAxEAPwDl6KKK5j9BP//Z",
        validate=True,
    )
    if not data.startswith(b"\xff\xd8\xff") or not data.endswith(b"\xff\xd9"):
        raise RuntimeError("Embedded JPEG test fixture is invalid.")
    return data


def main() -> int:
    print("=== Disease Akinator Contribution Integration Test ===")

    try:
        png_bytes = make_test_png()
        jpeg_bytes = make_test_jpeg()
    except Exception as exc:
        print(f"SETUP ERROR: Embedded image fixtures are invalid: {type(exc).__name__}")
        return 1

    print(
        f"PASS: Standard-library test fixtures ready "
        f"(PNG {len(png_bytes)} bytes, JPEG {len(jpeg_bytes)} bytes)"
    )

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
        IMAGE_BUCKET,
        DEGREE_BUCKET,
        delete_file,
        download_file,
        get_file_metadata,
    )
    from database.postgres import (
        create_image_contribution,
        create_symptom_contribution,
        delete_doctor,
        delete_image_contribution,
        delete_symptom_contribution,
        get_contribution_stats,
        get_image_contribution,
        get_symptom_contribution,
        list_image_contributions,
        list_symptom_contributions,
    )
    from services.doctor_service import register_doctor
    from services.contribution_service import (
        submit_image_contribution,
        submit_symptom_contribution,
    )

    run_id = uuid4().hex
    email = f"phase5-contrib-{run_id}@example.invalid"
    doctor_password = f"Pa9!{uuid4().hex}Z"
    pdf_bytes = b"%PDF-1.4\nTemporary contribution-test doctor PDF.\n%%EOF\n"

    doctor_id = None
    degree_file_id = None
    symptom_id = None
    image_rows: list[dict] = []
    image_file_ids: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        print(f"PASS: {label}")

    before = None
    try:
        before = get_contribution_stats()
        doctor = register_doctor(
            full_name="Temporary Contribution Test Doctor",
            age=35,
            degree="MBBS",
            custom_degree=None,
            hospital="Temporary Contribution Clinic",
            city="Pune",
            email=email,
            password=doctor_password,
            degree_filename="contribution-test-degree.pdf",
            degree_file_bytes=pdf_bytes,
        )
        doctor_id = doctor["id"]
        degree_file_id = doctor["degree_file_id"]
        check(True, "Create temporary doctor for contribution tests")

        ok, message = submit_symptom_contribution(
            doctor_id=doctor_id,
            disease_name="Phase 5 Test Disease",
            symptoms_list=["Fever", "Headache", "Joint pain", " fever "],
            notes="Temporary integration-test contribution",
        )
        check(ok, f"Submit symptom contribution ({message})")

        symptom_rows = list_symptom_contributions(doctor_id)
        matching_symptom = next(
            (row for row in symptom_rows if row["disease_name"] == "Phase 5 Test Disease"),
            None,
        )
        check(
            matching_symptom is not None
            and matching_symptom["doctor_id"] == doctor_id
            and matching_symptom["uploaded_by"] == "Temporary Contribution Test Doctor"
            and matching_symptom["symptoms"] == ["Fever", "Headache", "Joint pain"],
            "Neon stores symptoms, ordering, de-duplication and uploader identity",
        )
        symptom_id = matching_symptom["id"]

        symptom_detail = get_symptom_contribution(symptom_id)
        check(
            symptom_detail is not None
            and symptom_detail["symptoms"] == ["Fever", "Headache", "Joint pain"],
            "Read individual symptom contribution and child rows",
        )

        ok, message = submit_image_contribution(
            doctor_id=doctor_id,
            disease_name="Phase 5 Test Image Disease",
            image_bytes=png_bytes,
            image_name=f"phase5-{run_id}.png",
            notes="Synthetic PNG integration test",
        )
        check(ok, f"Submit PNG contribution ({message})")
        png_row = next(
            (
                row for row in list_image_contributions(doctor_id)
                if row["image_filename"] == f"phase5-{run_id}.png"
            ),
            None,
        )
        check(png_row is not None, "Find PostgreSQL metadata for uploaded PNG")
        image_rows.append(png_row)
        image_file_ids.append(png_row["mongo_file_id"])

        ok, message = submit_image_contribution(
            doctor_id=doctor_id,
            disease_name="Phase 5 Test JPEG Disease",
            image_bytes=jpeg_bytes,
            image_name=f"phase5-{run_id}.jpg",
            notes="Synthetic JPEG integration test",
        )
        check(ok, f"Submit JPEG contribution ({message})")
        jpeg_row = next(
            (
                row for row in list_image_contributions(doctor_id)
                if row["image_filename"] == f"phase5-{run_id}.jpg"
            ),
            None,
        )
        check(jpeg_row is not None, "Find PostgreSQL metadata for uploaded JPEG")
        image_rows.append(jpeg_row)
        image_file_ids.append(jpeg_row["mongo_file_id"])

        check(
            len(image_rows) == 2
            and all(
                row["doctor_id"] == doctor_id
                and row["uploaded_by"] == "Temporary Contribution Test Doctor"
                for row in image_rows
            ),
            "Neon image metadata identifies the contributing doctor",
        )

        expected_by_name = {
            f"phase5-{run_id}.png": (png_bytes, "image/png"),
            f"phase5-{run_id}.jpg": (jpeg_bytes, "image/jpeg"),
        }
        for row in image_rows:
            expected_bytes, expected_content_type = expected_by_name[row["image_filename"]]
            metadata = get_file_metadata(IMAGE_BUCKET, row["mongo_file_id"])
            check(
                metadata is not None
                and metadata["metadata"]["uploaded_by_doctor_id"] == doctor_id
                and metadata["metadata"]["content_type"] == expected_content_type,
                f"GridFS metadata records uploader and type for {row['image_filename']}",
            )
            check(
                download_file(IMAGE_BUCKET, row["mongo_file_id"]) == expected_bytes,
                f"GridFS download matches original {row['image_filename']} bytes",
            )

        after = get_contribution_stats()
        check(
            after["doctors"] == before["doctors"] + 1
            and after["symptom_entries"] == before["symptom_entries"] + 1
            and after["image_entries"] == before["image_entries"] + 2,
            "PostgreSQL contribution statistics count the new records",
        )

        print("\n--- Cleanup ---")
        for row in image_rows:
            check(
                delete_image_contribution(row["id"]),
                f"Delete PostgreSQL image metadata {row['id']}",
            )
        image_rows.clear()

        for file_id in image_file_ids:
            check(
                delete_file(IMAGE_BUCKET, file_id),
                "Delete GridFS contribution image",
            )
        image_file_ids.clear()

        check(
            delete_symptom_contribution(symptom_id),
            "Delete symptom contribution and cascade symptom items",
        )
        symptom_id = None

        check(delete_doctor(doctor_id), "Delete temporary doctor")
        doctor_id = None

        check(
            delete_file(DEGREE_BUCKET, degree_file_id),
            "Delete temporary doctor's degree PDF",
        )
        degree_file_id = None

        check(
            get_contribution_stats() == before,
            "All temporary contribution records removed and counts restored",
        )

        print("\n==============================================")
        print("ALL CONTRIBUTION INTEGRATION TESTS PASSED")
        print("==============================================")
        return 0

    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        return 1

    finally:
        # Discover test-child records by the isolated temporary doctor ID so
        # cleanup still works if an assertion fails before IDs are recorded.
        if doctor_id is not None:
            try:
                for row in list_image_contributions(doctor_id):
                    if row["image_filename"].startswith(f"phase5-{run_id}."):
                        delete_image_contribution(row["id"])
                        delete_file(IMAGE_BUCKET, row["mongo_file_id"])
            except Exception:
                print("WARNING: Image contribution cleanup may be required.")

            # Also delete known file IDs in case their SQL metadata was already
            # removed before a later cleanup assertion failed.
            for file_id in image_file_ids:
                try:
                    delete_file(IMAGE_BUCKET, file_id)
                except Exception:
                    print("WARNING: Orphaned GridFS file cleanup may be required.")

            try:
                for row in list_symptom_contributions(doctor_id):
                    if row["disease_name"] == "Phase 5 Test Disease":
                        delete_symptom_contribution(row["id"])
            except Exception:
                print("WARNING: Symptom contribution cleanup may be required.")

            try:
                delete_doctor(doctor_id)
            except Exception:
                print("WARNING: Test doctor cleanup may be required.")

        if degree_file_id is not None:
            try:
                delete_file(DEGREE_BUCKET, degree_file_id)
            except Exception:
                print("WARNING: Degree PDF cleanup may be required.")


if __name__ == "__main__":
    raise SystemExit(main())
