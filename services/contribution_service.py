"""Doctor contribution orchestration for Disease Akinator.

Structured symptom and contribution metadata lives in Neon PostgreSQL.
Contribution image binaries live in MongoDB Atlas GridFS.

The submit functions intentionally preserve the existing db_utils.py
contract: they return (success: bool, message: str), so app.py can be wired
to this service with minimal UI changes in Phase 5.3.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from database.mongodb import (
    IMAGE_BUCKET,
    delete_file,
    upload_contribution_image,
)
from database.postgres import (
    create_image_contribution,
    create_symptom_contribution,
    get_contribution_stats,
)


logger = logging.getLogger(__name__)


def _validate_doctor_id(doctor_id: int) -> None:
    if (
        isinstance(doctor_id, bool)
        or not isinstance(doctor_id, int)
        or doctor_id <= 0
    ):
        raise ValueError("A valid doctor account is required.")


def _validate_disease_name(disease_name: str) -> str:
    if not isinstance(disease_name, str) or not disease_name.strip():
        raise ValueError("Please enter a disease name.")
    disease = disease_name.strip()
    if len(disease) > 200:
        raise ValueError("Disease name must be 200 characters or fewer.")
    return disease


def _normalise_notes(notes: str | None) -> str | None:
    if notes is None:
        return None
    if not isinstance(notes, str):
        raise ValueError("Clinical notes must be text.")
    return notes.strip() or None


def submit_symptom_contribution(
    doctor_id: int,
    disease_name: str,
    symptoms_list: Sequence[str],
    notes: str = "",
) -> tuple[bool, str]:
    """Save a symptom-to-disease contribution in Neon PostgreSQL."""
    try:
        _validate_doctor_id(doctor_id)
        disease = _validate_disease_name(disease_name)
        normalized_notes = _normalise_notes(notes)

        create_symptom_contribution(
            doctor_id=doctor_id,
            disease_name=disease,
            symptoms=symptoms_list,
            notes=normalized_notes,
        )
        return True, "Symptom data submitted successfully!"

    except (TypeError, ValueError) as exc:
        return False, str(exc)
    except Exception:
        logger.exception(
            "Failed to save symptom contribution for doctor_id=%s",
            doctor_id if isinstance(doctor_id, int) else "invalid",
        )
        return False, "Unable to save symptom data right now. Please try again."


def submit_image_contribution(
    doctor_id: int,
    disease_name: str,
    image_bytes: bytes | bytearray | memoryview,
    image_name: str,
    notes: str = "",
) -> tuple[bool, str]:
    """Upload an image to GridFS, then store its metadata in Neon.

    If PostgreSQL metadata insertion fails after a successful GridFS upload,
    the uploaded GridFS file is deleted as a compensating action. If that
    cleanup also fails, the file ID is written to the application log for
    operator recovery; the user receives a non-sensitive error message.
    """
    uploaded_file_id: str | None = None

    try:
        _validate_doctor_id(doctor_id)
        disease = _validate_disease_name(disease_name)
        normalized_notes = _normalise_notes(notes)

        uploaded = upload_contribution_image(
            file_bytes=image_bytes,
            filename=image_name,
            uploaded_by_doctor_id=doctor_id,
        )
        uploaded_file_id = uploaded["file_id"]

        create_image_contribution(
            doctor_id=doctor_id,
            disease_name=disease,
            mongo_file_id=uploaded_file_id,
            image_filename=uploaded["filename"],
            content_type=uploaded["content_type"],
            file_size_bytes=uploaded["file_size_bytes"],
            notes=normalized_notes,
        )
        return True, "Image submitted successfully!"

    except (TypeError, ValueError) as exc:
        return False, str(exc)

    except Exception:
        logger.exception(
            "Failed to save image contribution for doctor_id=%s",
            doctor_id if isinstance(doctor_id, int) else "invalid",
        )

        if uploaded_file_id is not None:
            try:
                removed = delete_file(IMAGE_BUCKET, uploaded_file_id)
                if not removed:
                    logger.error(
                        "GridFS compensation could not confirm deletion; "
                        "bucket=%s file_id=%s",
                        IMAGE_BUCKET,
                        uploaded_file_id,
                    )
            except Exception:
                logger.exception(
                    "GridFS compensation failed; manual cleanup may be needed; "
                    "bucket=%s file_id=%s",
                    IMAGE_BUCKET,
                    uploaded_file_id,
                )
                return (
                    False,
                    "The image could not be linked to its database record, "
                    "and cleanup could not be confirmed. Please contact the "
                    "project administrator before retrying.",
                )

        return False, "Unable to save the image right now. Please try again."


def get_doctor_contribution_stats() -> dict[str, int]:
    """Return totals used by the existing Doctor Portal dashboard."""
    return get_contribution_stats()
