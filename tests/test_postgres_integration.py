"""Manual PostgreSQL integration test for Disease Akinator.

Run from the repository root:
    python tests/test_postgres_integration.py

The test asks for the Neon application URI and password without echoing
either credential. It creates uniquely named test records and removes them.
Run only against the dedicated Disease Akinator test database.
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from psycopg.conninfo import make_conninfo
from psycopg.errors import ForeignKeyViolation

# When invoked as "python tests\\test_postgres_integration.py", Python puts
# the tests directory (not the repository root) at sys.path[0]. Add the
# repository root so the sibling "database" package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

POSTGRES_MODULE = PROJECT_ROOT / "database" / "postgres.py"
DATABASE_INIT = PROJECT_ROOT / "database" / "__init__.py"


def main() -> int:
    print("=== Disease Akinator PostgreSQL Integration Test ===")

    # Fail early with a useful setup message instead of asking for credentials
    # and then raising an opaque ModuleNotFoundError.
    if not DATABASE_INIT.is_file() or not POSTGRES_MODULE.is_file():
        print("SETUP ERROR: The local database package is missing.")
        print(f"Expected: {DATABASE_INIT}")
        print(f"Expected: {POSTGRES_MODULE}")
        print("Download database/__init__.py and database/postgres.py")
        print("from the feature/database-migration branch, or extract")
        print("the supplied Phase 2 test bundle into this folder.")
        return 1

    uri = getpass.getpass(
        "Paste Neon application URI (hidden): "
    ).strip()
    parsed = urlsplit(uri)

    if unquote(parsed.username or "") != "disease_akinator_app":
        print("FAIL: URI must use disease_akinator_app.")
        return 1
    if parsed.path.lstrip("/").split("?")[0] != "disease_akinator":
        print("FAIL: URI must target disease_akinator.")
        return 1

    password = getpass.getpass("Application password (hidden): ")
    if not password:
        print("FAIL: Password cannot be empty.")
        return 1

    # Override any URI password safely. Nothing is printed or written to disk.
    os.environ["DISEASE_AKINATOR_POSTGRES_URL"] = make_conninfo(
        uri, password=password
    )

    from database.postgres import (
        check_database_connection,
        create_doctor,
        get_doctor_by_email,
        get_doctor_auth_record,
        get_doctor_by_id,
        update_doctor_profile,
        delete_doctor,
        create_symptom_contribution,
        get_symptom_contribution,
        list_symptom_contributions,
        update_symptom_contribution,
        delete_symptom_contribution,
        create_image_contribution,
        get_image_contribution,
        list_image_contributions,
        update_image_contribution,
        delete_image_contribution,
        get_contribution_stats,
    )

    run_id = uuid4().hex
    email = f"phase2-{run_id}@example.invalid"
    degree_file_id = run_id[:24]
    image_file_id = uuid4().hex[:24]

    doctor_id = None
    symptom_id = None
    image_id = None
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        print(f"PASS: {label}")

    try:
        identity = check_database_connection()
        check(
            identity["database_user"] == "disease_akinator_app"
            and identity["database_name"] == "disease_akinator",
            "Connection uses the restricted application role and target database",
        )

        before = get_contribution_stats()

        doctor = create_doctor(
            full_name="Phase 2 Integration Test",
            age=35,
            degree="MBBS",
            custom_degree=None,
            hospital="Temporary Test Clinic",
            city="Pune",
            email=email.upper(),
            password_hash="phase2-test-hash-not-for-authentication",
            degree_file_id=degree_file_id,
            degree_filename="phase2-test-degree.pdf",
            degree_size_bytes=1,
        )
        doctor_id = doctor["id"]
        check(doctor["email"] == email, "Create doctor and normalize email")

        check(
            get_doctor_by_email(email.upper())["id"] == doctor_id,
            "Read doctor by email",
        )
        check(
            get_doctor_by_id(doctor_id)["id"] == doctor_id,
            "Read doctor by ID",
        )

        auth_record = get_doctor_auth_record(email)
        check(
            auth_record["password_hash"]
            == "phase2-test-hash-not-for-authentication",
            "Retrieve authentication hash only through dedicated auth lookup",
        )
        check(
            "password_hash" not in get_doctor_by_id(doctor_id),
            "Normal doctor profile lookup does not expose password hash",
        )

        updated_doctor = update_doctor_profile(
            doctor_id,
            full_name="Phase 2 Integration Test Updated",
            age=36,
            degree="MD",
            custom_degree=None,
            hospital="Updated Test Clinic",
            city="Mumbai",
        )
        check(
            updated_doctor is not None
            and updated_doctor["city"] == "Mumbai",
            "Update doctor profile",
        )

        symptom = create_symptom_contribution(
            doctor_id,
            "Phase 2 Test Disease",
            ["Fever", "Headache", "Joint pain", " fever "],
            "Temporary integration-test notes",
        )
        symptom_id = symptom["id"]
        check(
            symptom["symptoms"] == ["Fever", "Headache", "Joint pain"],
            "Create symptom contribution and de-duplicate symptoms",
        )

        fetched_symptom = get_symptom_contribution(symptom_id)
        check(
            fetched_symptom is not None
            and fetched_symptom["doctor_id"] == doctor_id
            and len(fetched_symptom["symptoms"]) == 3,
            "Read symptom contribution and uploader doctor ID",
        )

        symptom_list = list_symptom_contributions(doctor_id)
        check(
            any(item["id"] == symptom_id for item in symptom_list),
            "List symptom contributions filtered by uploader",
        )

        updated_symptom = update_symptom_contribution(
            symptom_id,
            disease_name="Updated Phase 2 Test Disease",
            symptoms=["Fever", "Rash"],
            notes="Updated test notes",
        )
        check(
            updated_symptom is not None
            and updated_symptom["symptoms"] == ["Fever", "Rash"],
            "Update symptom contribution and symptom items atomically",
        )

        # Verify the doctor cannot be deleted while contributions reference it.
        try:
            delete_doctor(doctor_id)
        except ForeignKeyViolation:
            print("PASS: Foreign key prevents deleting a doctor with contributions")
        else:
            raise AssertionError(
                "Doctor deletion should have been blocked by contribution foreign keys"
            )

        image = create_image_contribution(
            doctor_id=doctor_id,
            disease_name="Phase 2 Test Image Disease",
            mongo_file_id=image_file_id,
            image_filename="phase2-test.jpg",
            content_type="image/jpeg",
            file_size_bytes=1,
            notes="Metadata-only PostgreSQL integration test",
        )
        image_id = image["id"]
        check(
            image["doctor_id"] == doctor_id
            and image["mongo_file_id"] == image_file_id,
            "Create image metadata with uploader and MongoDB file reference",
        )

        fetched_image = get_image_contribution(image_id)
        check(
            fetched_image is not None
            and fetched_image["uploaded_by"] == "Phase 2 Integration Test Updated",
            "Read image metadata and uploader identity",
        )

        image_list = list_image_contributions(doctor_id)
        check(
            any(item["id"] == image_id for item in image_list),
            "List image contributions filtered by uploader",
        )

        updated_image = update_image_contribution(
            image_id,
            disease_name="Updated Phase 2 Test Image Disease",
            notes="Updated metadata test",
        )
        check(
            updated_image is not None
            and updated_image["disease_name"]
            == "Updated Phase 2 Test Image Disease",
            "Update image contribution metadata",
        )

        after = get_contribution_stats()
        check(
            after["doctors"] == before["doctors"] + 1
            and after["symptom_entries"] == before["symptom_entries"] + 1
            and after["image_entries"] == before["image_entries"] + 1,
            "Contribution statistics reflect inserted records",
        )

        check(
            delete_symptom_contribution(symptom_id),
            "Delete symptom contribution (child symptoms cascade)",
        )
        symptom_id = None

        check(delete_image_contribution(image_id), "Delete image contribution metadata")
        image_id = None

        check(delete_doctor(doctor_id), "Delete test doctor after contributions are removed")
        doctor_id = None

        final_stats = get_contribution_stats()
        check(final_stats == before, "All test records removed; statistics restored")

        print("\n==============================================")
        print("ALL POSTGRESQL INTEGRATION TESTS PASSED")
        print("==============================================")
        return 0

    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"\nFAIL: {failures[-1]}")
        return 1

    finally:
        # Best-effort cleanup even if a test assertion fails.
        try:
            if symptom_id is not None:
                delete_symptom_contribution(symptom_id)
        except Exception as exc:
            print(f"WARNING: Symptom cleanup required: {type(exc).__name__}")

        try:
            if image_id is not None:
                delete_image_contribution(image_id)
        except Exception as exc:
            print(f"WARNING: Image metadata cleanup required: {type(exc).__name__}")

        try:
            if doctor_id is not None:
                delete_doctor(doctor_id)
        except Exception as exc:
            print(f"WARNING: Doctor cleanup required: {type(exc).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
