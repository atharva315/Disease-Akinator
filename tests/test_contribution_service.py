"""Unit tests for doctor contribution orchestration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.contribution_service import (
    get_doctor_contribution_stats,
    submit_image_contribution,
    submit_symptom_contribution,
)


class SymptomContributionTests(unittest.TestCase):
    @patch("services.contribution_service.create_symptom_contribution")
    def test_successful_symptom_submission_preserves_existing_contract(self, mock_create):
        mock_create.return_value = {"id": 10}

        result = submit_symptom_contribution(
            doctor_id=12,
            disease_name="  Dengue  ",
            symptoms_list=["Fever", "Headache"],
            notes="  Clinical note  ",
        )

        self.assertEqual(result, (True, "Symptom data submitted successfully!"))
        mock_create.assert_called_once_with(
            doctor_id=12,
            disease_name="Dengue",
            symptoms=["Fever", "Headache"],
            notes="Clinical note",
        )

    @patch("services.contribution_service.create_symptom_contribution")
    def test_invalid_symptoms_are_returned_as_validation_message(self, mock_create):
        mock_create.side_effect = ValueError("At least two distinct symptoms are required.")

        result = submit_symptom_contribution(
            doctor_id=12,
            disease_name="Dengue",
            symptoms_list=["Fever"],
        )

        self.assertEqual(
            result,
            (False, "At least two distinct symptoms are required."),
        )
        mock_create.assert_called_once()

    @patch("services.contribution_service.create_symptom_contribution")
    def test_invalid_doctor_id_stops_before_database_call(self, mock_create):
        result = submit_symptom_contribution(
            doctor_id=True,
            disease_name="Dengue",
            symptoms_list=["Fever", "Headache"],
        )

        self.assertFalse(result[0])
        self.assertIn("doctor account", result[1])
        mock_create.assert_not_called()

    @patch("services.contribution_service.create_symptom_contribution")
    def test_database_failure_returns_safe_message(self, mock_create):
        mock_create.side_effect = RuntimeError("sensitive connection detail")

        result = submit_symptom_contribution(
            doctor_id=12,
            disease_name="Dengue",
            symptoms_list=["Fever", "Headache"],
        )

        self.assertEqual(
            result,
            (False, "Unable to save symptom data right now. Please try again."),
        )
        self.assertNotIn("sensitive", result[1])


class ImageContributionTests(unittest.TestCase):
    def setUp(self):
        self.uploaded = {
            "file_id": "0123456789abcdef01234567",
            "bucket": "contribution_images",
            "filename": "rash.png",
            "content_type": "image/png",
            "file_size_bytes": 100,
        }

    @patch("services.contribution_service.create_image_contribution")
    @patch("services.contribution_service.upload_contribution_image")
    def test_successful_image_submission_links_gridfs_and_postgres(
        self, mock_upload, mock_create
    ):
        mock_upload.return_value = self.uploaded
        mock_create.return_value = {"id": 77}

        result = submit_image_contribution(
            doctor_id=12,
            disease_name="  Melanoma  ",
            image_bytes=b"synthetic-image-bytes",
            image_name="rash.png",
            notes="  Image note  ",
        )

        self.assertEqual(result, (True, "Image submitted successfully!"))
        mock_upload.assert_called_once_with(
            file_bytes=b"synthetic-image-bytes",
            filename="rash.png",
            uploaded_by_doctor_id=12,
        )
        mock_create.assert_called_once_with(
            doctor_id=12,
            disease_name="Melanoma",
            mongo_file_id=self.uploaded["file_id"],
            image_filename="rash.png",
            content_type="image/png",
            file_size_bytes=100,
            notes="Image note",
        )

    @patch("services.contribution_service.delete_file")
    @patch("services.contribution_service.create_image_contribution")
    @patch("services.contribution_service.upload_contribution_image")
    def test_postgres_failure_deletes_uploaded_gridfs_file(
        self, mock_upload, mock_create, mock_delete
    ):
        mock_upload.return_value = self.uploaded
        mock_create.side_effect = RuntimeError("simulated Neon insert failure")
        mock_delete.return_value = True

        result = submit_image_contribution(
            doctor_id=12,
            disease_name="Melanoma",
            image_bytes=b"synthetic-image-bytes",
            image_name="rash.png",
        )

        self.assertFalse(result[0])
        self.assertIn("Unable to save the image", result[1])
        mock_delete.assert_called_once_with(
            "contribution_images",
            self.uploaded["file_id"],
        )

    @patch("services.contribution_service.delete_file")
    @patch("services.contribution_service.create_image_contribution")
    @patch("services.contribution_service.upload_contribution_image")
    def test_failed_compensation_returns_operator_attention_message(
        self, mock_upload, mock_create, mock_delete
    ):
        mock_upload.return_value = self.uploaded
        mock_create.side_effect = RuntimeError("simulated Neon insert failure")
        mock_delete.side_effect = RuntimeError("simulated Atlas delete failure")

        result = submit_image_contribution(
            doctor_id=12,
            disease_name="Melanoma",
            image_bytes=b"synthetic-image-bytes",
            image_name="rash.png",
        )

        self.assertFalse(result[0])
        self.assertIn("cleanup could not be confirmed", result[1])

    @patch("services.contribution_service.delete_file")
    @patch("services.contribution_service.create_image_contribution")
    @patch("services.contribution_service.upload_contribution_image")
    def test_upload_failure_does_not_attempt_metadata_or_cleanup(
        self, mock_upload, mock_create, mock_delete
    ):
        mock_upload.side_effect = ValueError("Only PNG and JPEG are accepted.")

        result = submit_image_contribution(
            doctor_id=12,
            disease_name="Melanoma",
            image_bytes=b"invalid",
            image_name="rash.gif",
        )

        self.assertEqual(
            result,
            (False, "Only PNG and JPEG are accepted."),
        )
        mock_create.assert_not_called()
        mock_delete.assert_not_called()

    @patch("services.contribution_service.get_contribution_stats")
    def test_statistics_wrapper_returns_database_totals(self, mock_stats):
        expected = {"doctors": 2, "symptom_entries": 3, "image_entries": 4}
        mock_stats.return_value = expected

        self.assertEqual(get_doctor_contribution_stats(), expected)
        mock_stats.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
