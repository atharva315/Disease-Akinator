"""Unit tests for the doctor registration orchestration service."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.doctor_service import (
    DoctorRegistrationError,
    DuplicateDoctorEmailError,
    register_doctor,
)


PDF = b"%PDF-1.4\nSynthetic registration test PDF\n%%EOF\n"


class DoctorRegistrationUnitTests(unittest.TestCase):
    def setUp(self):
        self.valid = {
            "full_name": "  Dr. Example  ",
            "age": 35,
            "degree": "MBBS",
            "custom_degree": None,
            "hospital": " Example Clinic ",
            "city": " Pune ",
            "email": " DOCTOR@EXAMPLE.INVALID ",
            "password": "Correct Horse Battery 42!",
            "degree_filename": "degree.pdf",
            "degree_file_bytes": PDF,
        }
        self.profile = {
            "id": 801,
            "full_name": "Dr. Example",
            "email": "doctor@example.invalid",
            "degree": "MBBS",
            "degree_file_id": "0123456789abcdef01234567",
        }

    @patch("services.doctor_service.create_doctor")
    @patch("services.doctor_service.upload_degree_pdf")
    @patch("services.doctor_service.get_doctor_by_email")
    def test_successful_registration_hashes_password_and_links_pdf(
        self, mock_existing, mock_upload, mock_create
    ):
        mock_existing.return_value = None
        mock_upload.return_value = {
            "file_id": "0123456789abcdef01234567",
            "filename": "degree.pdf",
            "file_size_bytes": len(PDF),
        }
        mock_create.return_value = self.profile

        result = register_doctor(**self.valid)

        self.assertEqual(result, self.profile)
        mock_existing.assert_called_once_with("doctor@example.invalid")
        upload_args = mock_upload.call_args.args
        self.assertEqual(upload_args[0], PDF)
        self.assertEqual(upload_args[1], "degree.pdf")
        self.assertEqual(upload_args[2], "doctor@example.invalid")

        create_kwargs = mock_create.call_args.kwargs
        self.assertEqual(create_kwargs["email"], "doctor@example.invalid")
        self.assertTrue(create_kwargs["password_hash"].startswith("$argon2id$"))
        self.assertNotEqual(
            create_kwargs["password_hash"], self.valid["password"]
        )
        self.assertEqual(
            create_kwargs["degree_file_id"],
            "0123456789abcdef01234567",
        )

    @patch("services.doctor_service.upload_degree_pdf")
    @patch("services.doctor_service.get_doctor_by_email")
    def test_duplicate_email_stops_before_file_upload(
        self, mock_existing, mock_upload
    ):
        mock_existing.return_value = {"id": 10, "email": "doctor@example.invalid"}

        with self.assertRaises(DuplicateDoctorEmailError):
            register_doctor(**self.valid)

        mock_upload.assert_not_called()

    @patch("services.doctor_service.delete_file")
    @patch("services.doctor_service.create_doctor")
    @patch("services.doctor_service.upload_degree_pdf")
    @patch("services.doctor_service.get_doctor_by_email")
    def test_database_failure_compensates_by_deleting_pdf(
        self, mock_existing, mock_upload, mock_create, mock_delete
    ):
        mock_existing.return_value = None
        mock_upload.return_value = {
            "file_id": "0123456789abcdef01234567",
            "filename": "degree.pdf",
            "file_size_bytes": len(PDF),
        }
        mock_create.side_effect = RuntimeError("simulated Neon insert failure")

        with self.assertRaises(DoctorRegistrationError):
            register_doctor(**self.valid)

        mock_delete.assert_called_once_with(
            "doctor_degrees",
            "0123456789abcdef01234567",
        )

    @patch("services.doctor_service.delete_file")
    @patch("services.doctor_service.create_doctor")
    @patch("services.doctor_service.upload_degree_pdf")
    @patch("services.doctor_service.get_doctor_by_email")
    def test_reports_uncertain_cleanup_if_compensation_fails(
        self, mock_existing, mock_upload, mock_create, mock_delete
    ):
        mock_existing.return_value = None
        mock_upload.return_value = {
            "file_id": "0123456789abcdef01234567",
            "filename": "degree.pdf",
            "file_size_bytes": len(PDF),
        }
        mock_create.side_effect = RuntimeError("simulated Neon failure")
        mock_delete.side_effect = RuntimeError("simulated Atlas failure")

        with self.assertRaisesRegex(
            DoctorRegistrationError, "cleanup could not be confirmed"
        ):
            register_doctor(**self.valid)

    def test_rejects_non_pdf_filename(self):
        values = dict(self.valid, degree_filename="degree.png")
        with self.assertRaisesRegex(ValueError, "must be a PDF"):
            register_doctor(**values)

    def test_rejects_pdf_extension_with_non_pdf_content(self):
        values = dict(self.valid, degree_file_bytes=b"not really a PDF")
        with self.assertRaisesRegex(ValueError, "not a valid PDF"):
            register_doctor(**values)

    def test_rejects_degree_other_without_custom_value(self):
        values = dict(self.valid, degree="Other", custom_degree=" ")
        with self.assertRaisesRegex(ValueError, "Custom degree"):
            register_doctor(**values)

    def test_accepts_other_with_custom_value(self):
        values = dict(
            self.valid,
            degree="Other",
            custom_degree="Specialist Diploma",
        )
        with patch("services.doctor_service.get_doctor_by_email", return_value=None), \
             patch("services.doctor_service.upload_degree_pdf", return_value={
                 "file_id": "0123456789abcdef01234567",
                 "filename": "degree.pdf",
                 "file_size_bytes": len(PDF),
             }), \
             patch("services.doctor_service.create_doctor", return_value=self.profile) as mock_create:
            register_doctor(**values)
            self.assertEqual(
                mock_create.call_args.kwargs["custom_degree"],
                "Specialist Diploma",
            )

    def test_rejects_short_password_before_database_calls(self):
        values = dict(self.valid, password="short")
        with patch("services.doctor_service.get_doctor_by_email") as mock_existing:
            with self.assertRaisesRegex(ValueError, "at least 12"):
                register_doctor(**values)
            mock_existing.assert_not_called()

    def test_rejects_invalid_email(self):
        values = dict(self.valid, email="not-an-email")
        with patch("services.doctor_service.get_doctor_by_email") as mock_existing:
            with self.assertRaisesRegex(ValueError, "valid email"):
                register_doctor(**values)
            mock_existing.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
