"""Real Neon integration test for Argon2id doctor authentication.

Run from the repository root:
    python tests/test_auth_integration.py

The script prompts for the restricted Neon URI/password without echoing them,
creates one temporary doctor using an Argon2id hash, verifies correct and
incorrect login, then deletes the temporary doctor.
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
    print("=== Disease Akinator Argon2id Authentication Integration Test ===")

    uri = getpass.getpass("Paste Neon application URI (hidden): ").strip()
    parsed = urlsplit(uri)

    if unquote(parsed.username or "") != "disease_akinator_app":
        print("FAIL: URI must use disease_akinator_app.")
        return 1
    if parsed.path.lstrip("/").split("?")[0] != "disease_akinator":
        print("FAIL: URI must target disease_akinator.")
        return 1

    password_for_database = getpass.getpass(
        "Neon application database password (hidden): "
    )
    if not password_for_database:
        print("FAIL: Database password cannot be empty.")
        return 1

    os.environ["DISEASE_AKINATOR_POSTGRES_URL"] = make_conninfo(
        uri, password=password_for_database
    )

    from database.postgres import (
        create_doctor,
        delete_doctor,
        get_doctor_auth_record,
    )
    from services.auth_service import (
        authenticate_doctor,
        hash_password,
    )

    run_id = uuid4().hex
    email = f"auth-phase4-{run_id}@example.invalid"
    raw_password = f"Aa9!{uuid4().hex}Z"
    doctor_id = None

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        print(f"PASS: {label}")

    try:
        password_hash = hash_password(raw_password)
        check(
            password_hash.startswith("$argon2id$"),
            "New doctor password is stored as an Argon2id hash",
        )

        doctor = create_doctor(
            full_name="Temporary Authentication Test Doctor",
            age=35,
            degree="MBBS",
            custom_degree=None,
            hospital="Temporary Authentication Test Clinic",
            city="Pune",
            email=email,
            password_hash=password_hash,
            degree_file_id=run_id[:24],
            degree_filename="auth-test-degree.pdf",
            degree_size_bytes=100,
        )
        doctor_id = doctor["id"]
        check(doctor["email"] == email, "Create temporary doctor account")

        stored_auth = get_doctor_auth_record(email)
        check(
            stored_auth is not None
            and stored_auth["password_hash"].startswith("$argon2id$"),
            "Neon stores the Argon2id hash, not the plaintext password",
        )

        profile = authenticate_doctor(email.upper(), raw_password)
        check(
            profile is not None and profile["id"] == doctor_id,
            "Correct email and password authenticate successfully",
        )
        check(
            profile is not None and "password_hash" not in profile,
            "Authenticated profile does not expose password hash",
        )

        invalid = authenticate_doctor(email, "Definitely incorrect 123!")
        check(
            invalid is None,
            "Incorrect password is rejected without returning a profile",
        )

        missing = authenticate_doctor(
            f"missing-{run_id}@example.invalid",
            raw_password,
        )
        check(missing is None, "Unknown email is rejected")

        print("\n==============================================")
        print("ALL ARGON2ID AUTHENTICATION TESTS PASSED")
        print("==============================================")
        return 0

    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        return 1

    finally:
        if doctor_id is not None:
            try:
                if delete_doctor(doctor_id):
                    print("Cleanup: temporary authentication test doctor removed.")
            except Exception as cleanup_error:
                print(
                    "WARNING: Manual test-doctor cleanup may be required: "
                    f"{type(cleanup_error).__name__}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
