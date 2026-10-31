"""Doctor password authentication service for Disease Akinator.

This module contains password hashing and login verification only.
Streamlit UI wiring and registration orchestration are handled in the
later doctor-portal integration phase.
"""

from __future__ import annotations

from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)

from database.postgres import get_doctor_auth_record, get_doctor_by_id


_PASSWORD_HASHER = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash a new doctor password with Argon2id.

    Passwords are not stripped or normalized: leading/trailing spaces remain
    part of the password. A minimum of 12 characters is required.
    """
    if not isinstance(password, str):
        raise TypeError("Password must be a string.")
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    if len(password) > 1024:
        raise ValueError("Password is too long.")

    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Verify a password against its Argon2id hash without raising on mismatch."""
    if not isinstance(password_hash, str) or not password_hash:
        return False
    if not isinstance(password, str):
        return False

    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def authenticate_doctor(email: str, password: str) -> dict[str, Any] | None:
    """Authenticate a doctor and return a profile without its password hash.

    Returns None for an unknown email or incorrect password. Database
    connectivity/configuration errors are allowed to propagate so the caller
    can distinguish an unavailable database from invalid credentials.
    """
    if not isinstance(email, str) or not email.strip():
        return None
    if not isinstance(password, str) or not password:
        return None

    normalized_email = email.strip().lower()
    auth_record = get_doctor_auth_record(normalized_email)

    if auth_record is None:
        return None

    stored_hash = auth_record.get("password_hash")
    if not verify_password(stored_hash, password):
        return None

    # Fetch the standard safe profile; this query intentionally excludes
    # password_hash, so the credential hash is never returned to Streamlit state.
    return get_doctor_by_id(auth_record["id"])


def password_needs_rehash(password_hash: str) -> bool:
    """Indicate whether a stored hash should be upgraded to current parameters."""
    if not isinstance(password_hash, str) or not password_hash:
        return True

    try:
        return _PASSWORD_HASHER.check_needs_rehash(password_hash)
    except (VerificationError, InvalidHashError):
        return True
