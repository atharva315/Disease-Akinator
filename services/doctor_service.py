"""Doctor registration orchestration for Disease Akinator.

Coordinates Argon2id hashing, degree-PDF storage in MongoDB GridFS, and the
structured doctor record in Neon PostgreSQL. Cross-database failure handling
removes the newly uploaded PDF when the PostgreSQL insert is rejected.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from psycopg.errors import UniqueViolation

from database.mongodb import (
    MAX_DEGREE_BYTES,
    DEGREE_BUCKET,
    delete_file,
    upload_degree_pdf,
)
from database.postgres import create_doctor, get_doctor_by_email
from services.auth_service import hash_password


DEGREE_OPTIONS = frozenset(
    {
        "MBBS", "MD", "MS", "DM", "MCh", "BDS", "MDS", "DNB",
        "BAMS", "BHMS", "BUMS", "BSMS", "BNYS", "Other",
    }
)
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class DuplicateDoctorEmailError(ValueError):
    """Raised when a doctor account already exists for the email."""


class DoctorRegistrationError(RuntimeError):
    """Raised when the registration workflow cannot complete safely."""


def _required_text(value: str, label: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} is required.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} is required.")
    if len(cleaned) > max_length:
        raise ValueError(f"{label} must be {max_length} characters or fewer.")
    return cleaned


def _optional_text(value: str | None, label: str, max_length: int) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    cleaned = value.strip()
    if len(cleaned) > max_length:
        raise ValueError(f"{label} must be {max_length} characters or fewer.")
    return cleaned


def _validate_registration(
    *,
    full_name: str,
    age: int,
    degree: str,
    custom_degree: str | None,
    hospital: str | None,
    city: str | None,
    email: str,
    password: str,
    degree_filename: str,
    degree_file_bytes: bytes | bytearray | memoryview,
) -> dict[str, Any]:
    name = _required_text(full_name, "Full name", 100)

    if isinstance(age, bool):
        raise ValueError("Age must be between 25 and 80.")
    try:
        normalized_age = int(age)
    except (TypeError, ValueError) as exc:
        raise ValueError("Age must be a whole number between 25 and 80.") from exc
    if normalized_age != age or not 25 <= normalized_age <= 80:
        raise ValueError("Age must be between 25 and 80.")

    selected_degree = _required_text(degree, "Degree", 50)
    if selected_degree not in DEGREE_OPTIONS:
        raise ValueError("Please select a valid degree.")

    if selected_degree == "Other":
        normalized_custom_degree = _required_text(
            custom_degree or "", "Custom degree", 100
        )
    else:
        normalized_custom_degree = None

    normalized_email = _required_text(email, "Email", 254).lower()
    if not _EMAIL_PATTERN.fullmatch(normalized_email):
        raise ValueError("Please enter a valid email address.")

    # Argon2id service enforces the password length limits.
    if not isinstance(password, str):
        raise ValueError("Password is required.")
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    if len(password) > 1024:
        raise ValueError("Password is too long.")

    normalized_hospital = _optional_text(hospital, "Hospital / Clinic", 150)
    normalized_city = _optional_text(city, "City", 100)

    if not isinstance(degree_filename, str) or Path(degree_filename).suffix.lower() != ".pdf":
        raise ValueError("Doctor degree certificate must be a PDF file.")

    if not isinstance(degree_file_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("Please upload a PDF degree certificate.")
    file_bytes = bytes(degree_file_bytes)
    if not file_bytes:
        raise ValueError("The uploaded degree PDF is empty.")
    if len(file_bytes) > MAX_DEGREE_BYTES:
        raise ValueError("Degree PDF must be 10 MB or smaller.")
    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")

    return {
        "full_name": name,
        "age": normalized_age,
        "degree": selected_degree,
        "custom_degree": normalized_custom_degree,
        "hospital": normalized_hospital,
        "city": normalized_city,
        "email": normalized_email,
        "password": password,
        "degree_filename": Path(degree_filename).name,
        "degree_file_bytes": file_bytes,
    }


def register_doctor(
    *,
    full_name: str,
    age: int,
    degree: str,
    custom_degree: str | None,
    hospital: str | None,
    city: str | None,
    email: str,
    password: str,
    degree_filename: str,
    degree_file_bytes: bytes | bytearray | memoryview,
) -> dict[str, Any]:
    """Register a doctor and return the safe PostgreSQL profile.

    The GridFS degree is uploaded first. If the PostgreSQL insert fails, the
    newly uploaded GridFS file is deleted so an orphan is not left behind.
    """
    values = _validate_registration(
        full_name=full_name,
        age=age,
        degree=degree,
        custom_degree=custom_degree,
        hospital=hospital,
        city=city,
        email=email,
        password=password,
        degree_filename=degree_filename,
        degree_file_bytes=degree_file_bytes,
    )

    # Fast duplicate check avoids uploading a PDF for an already-registered
    # email. The database UNIQUE constraint remains the final concurrency guard.
    if get_doctor_by_email(values["email"]) is not None:
        raise DuplicateDoctorEmailError(
            "This email is already registered. Please log in instead."
        )

    password_hash = hash_password(values["password"])
    uploaded = upload_degree_pdf(
        values["degree_file_bytes"],
        values["degree_filename"],
        values["email"],
    )

    try:
        return create_doctor(
            full_name=values["full_name"],
            age=values["age"],
            degree=values["degree"],
            custom_degree=values["custom_degree"],
            hospital=values["hospital"],
            city=values["city"],
            email=values["email"],
            password_hash=password_hash,
            degree_file_id=uploaded["file_id"],
            degree_filename=uploaded["filename"],
            degree_size_bytes=uploaded["file_size_bytes"],
        )
    except UniqueViolation as exc:
        try:
            delete_file(DEGREE_BUCKET, uploaded["file_id"])
        except Exception as cleanup_exc:
            raise DoctorRegistrationError(
                "The email may already be registered, but PDF cleanup could "
                "not be confirmed. Check the account before retrying."
            ) from cleanup_exc

        constraint = getattr(getattr(exc, "diag", None), "constraint_name", None)
        if constraint == "doctors_email_unique":
            raise DuplicateDoctorEmailError(
                "This email is already registered. Please log in instead."
            ) from exc
        raise DoctorRegistrationError(
            "Registration could not be completed because of a database constraint."
        ) from exc
    except Exception as exc:
        try:
            delete_file(DEGREE_BUCKET, uploaded["file_id"])
        except Exception as cleanup_exc:
            raise DoctorRegistrationError(
                "Registration could not be completed and PDF cleanup could not "
                "be confirmed. Please contact the project administrator before retrying."
            ) from cleanup_exc
        raise DoctorRegistrationError(
            "Registration could not be completed. Please try again."
        ) from exc
