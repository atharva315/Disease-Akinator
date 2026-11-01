"""Unit tests for Disease Akinator doctor authentication.

These tests do not connect to Neon. Database calls are mocked so password
hashing and login behaviour can be verified locally and repeatedly.
Run:
    python -m unittest discover -s tests -p test_auth_service.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.auth_service import (
    authenticate_doctor,
    hash_password,
    password_needs_rehash,
    verify_password,
)


class PasswordHashingTests(unittest.TestCase):
    def test_hash_is_argon2id_and_not_plaintext(self):
        password = "Correct Horse Battery 42!"
        password_hash = hash_password(password)

        self.assertNotEqual(password_hash, password)
        self.assertTrue(password_hash.startswith("$argon2id$"))
        self.assertTrue(verify_password(password_hash, password))

    def test_wrong_password_is_rejected(self):
        password_hash = hash_password("Correct Horse Battery 42!")
        self.assertFalse(verify_password(password_hash, "Wrong password 123!"))

    def test_short_password_is_rejected_for_new_accounts(self):
        with self.assertRaises(ValueError):
            hash_password("short")

    def test_empty_password_is_rejected(self):
        with self.assertRaises(ValueError):
            hash_password("")

    def test_malformed_hash_does_not_authenticate(self):
        self.assertFalse(verify_password("not-an-argon2-hash", "Some password 123!"))

    def test_hash_rehash_check(self):
        password_hash = hash_password("Correct Horse Battery 42!")
        self.assertIsInstance(password_needs_rehash(password_hash), bool)


class DoctorAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.password = "Correct Horse Battery 42!"
        self.password_hash = hash_password(self.password)
        self.auth_record = {
            "id": 47,
            "full_name": "Test Doctor",
            "email": "doctor@example.invalid",
            "degree": "MBBS",
            "password_hash": self.password_hash,
        }
        self.profile = {
            "id": 47,
            "full_name": "Test Doctor",
            "age": 35,
            "degree": "MBBS",
            "custom_degree": None,
            "hospital": "Test Clinic",
            "city": "Pune",
            "email": "doctor@example.invalid",
            "degree_file_id": "0123456789abcdef01234567",
            "degree_filename": "degree.pdf",
            "degree_size_bytes": 100,
            "created_at": None,
        }

    @patch("services.auth_service.get_doctor_by_id")
    @patch("services.auth_service.get_doctor_auth_record")
    def test_successful_login_returns_safe_profile(
        self, mock_auth_record, mock_profile
    ):
        mock_auth_record.return_value = self.auth_record
        mock_profile.return_value = self.profile

        result = authenticate_doctor("  DOCTOR@EXAMPLE.INVALID ", self.password)

        mock_auth_record.assert_called_once_with("doctor@example.invalid")
        mock_profile.assert_called_once_with(47)
        self.assertEqual(result, self.profile)
        self.assertNotIn("password_hash", result)

    @patch("services.auth_service.get_doctor_by_id")
    @patch("services.auth_service.get_doctor_auth_record")
    def test_wrong_password_returns_none(self, mock_auth_record, mock_profile):
        mock_auth_record.return_value = self.auth_record

        result = authenticate_doctor(
            "doctor@example.invalid",
            "An incorrect password 456!",
        )

        self.assertIsNone(result)
        mock_profile.assert_not_called()

    @patch("services.auth_service.get_doctor_by_id")
    @patch("services.auth_service.get_doctor_auth_record")
    def test_unknown_email_returns_none(self, mock_auth_record, mock_profile):
        mock_auth_record.return_value = None

        result = authenticate_doctor(
            "missing@example.invalid",
            self.password,
        )

        self.assertIsNone(result)
        mock_profile.assert_not_called()

    def test_blank_email_returns_none_without_database_call(self):
        with patch("services.auth_service.get_doctor_auth_record") as mock_auth:
            self.assertIsNone(authenticate_doctor("   ", self.password))
            mock_auth.assert_not_called()

    def test_blank_password_returns_none_without_database_call(self):
        with patch("services.auth_service.get_doctor_auth_record") as mock_auth:
            self.assertIsNone(
                authenticate_doctor("doctor@example.invalid", "")
            )
            mock_auth.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
