"""Manual MongoDB Atlas/GridFS integration test for Disease Akinator.

Run from the repository root:
    python tests/test_mongodb_integration.py

Credentials are entered through hidden prompts and are not written to files.
Uploads uniquely generated minimal PDF/PNG/JPEG fixtures, downloads and
compares bytes, checks uploader metadata, rejects invalid formats, then
deletes every test file.
"""

from __future__ import annotations

import base64
import getpass
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, unquote
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_test_png() -> bytes:
    # A valid 1x1 RGB PNG (red pixel).
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg=="
    )


def make_test_jpeg() -> bytes:
    # Minimal valid JPEG fixture generated from a tiny known JPEG sample.
    return base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
        "//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////"
        "//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/"
        "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/"
        "xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA//2Q=="
    )


def main() -> int:
    print("=== Disease Akinator MongoDB/GridFS Integration Test ===")

    module_path = PROJECT_ROOT / "database" / "mongodb.py"
    if not module_path.is_file():
        print(f"SETUP ERROR: MongoDB module not found: {module_path}")
        print("Download database/mongodb.py from feature/database-migration.")
        return 1

    uri = getpass.getpass("Paste MongoDB Atlas application URI (hidden): ").strip()
    parsed = urlsplit(uri)
    if parsed.scheme not in {"mongodb", "mongodb+srv"}:
        print("FAIL: Invalid MongoDB URI scheme.")
        return 1
    if unquote(parsed.username or "") != "disease_akinator_app":
        print("FAIL: URI must use the restricted disease_akinator_app user.")
        return 1

    os.environ["DISEASE_AKINATOR_MONGODB_URI"] = uri

    from database.mongodb import (
        check_mongodb_connection,
        upload_degree_pdf,
        upload_contribution_image,
        download_file,
        get_file_metadata,
        delete_file,
        DEGREE_BUCKET,
        IMAGE_BUCKET,
    )

    run_id = uuid4().hex
    email = f"phase3-{run_id}@example.invalid"
    doctor_id = int(run_id[:7], 16) + 1

    uploaded: list[tuple[str, str]] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        print(f"PASS: {label}")

    # Minimal valid PDF header/body fixture. It is a storage-format test,
    # not a real or authentic medical degree certificate.
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [] /Count 0 >> endobj\n"
        b"trailer << /Root 1 0 R >>\n%%EOF\n"
    )
    png_bytes = make_test_png()
    jpeg_bytes = make_test_jpeg()

    try:
        connection = check_mongodb_connection()
        check(
            connection["connected"] is True
            and connection["database"] == "disease_akinator",
            "Connect and ping the Disease Akinator Atlas database",
        )

        degree = upload_degree_pdf(
            pdf_bytes,
            f"phase3-{run_id}.pdf",
            email,
        )
        uploaded.append((DEGREE_BUCKET, degree["file_id"]))
        check(
            len(degree["file_id"]) == 24
            and degree["content_type"] == "application/pdf",
            "Upload PDF into doctor_degrees",
        )

        downloaded_pdf = download_file(DEGREE_BUCKET, degree["file_id"])
        check(downloaded_pdf == pdf_bytes, "Download PDF and compare exact bytes")

        degree_meta = get_file_metadata(DEGREE_BUCKET, degree["file_id"])
        check(
            degree_meta is not None
            and degree_meta["metadata"]["uploaded_by_email"] == email
            and degree_meta["metadata"]["file_type"] == "doctor_degree",
            "Read PDF metadata and identify uploader",
        )

        image_png = upload_contribution_image(
            png_bytes,
            f"phase3-{run_id}.png",
            doctor_id,
        )
        uploaded.append((IMAGE_BUCKET, image_png["file_id"]))
        check(
            image_png["content_type"] == "image/png",
            "Upload PNG into contribution_images",
        )
        check(
            download_file(IMAGE_BUCKET, image_png["file_id"]) == png_bytes,
            "Download PNG and compare exact bytes",
        )
        png_meta = get_file_metadata(IMAGE_BUCKET, image_png["file_id"])
        check(
            png_meta is not None
            and png_meta["metadata"]["uploaded_by_doctor_id"] == doctor_id,
            "Read PNG metadata and identify uploader doctor ID",
        )

        image_jpeg = upload_contribution_image(
            jpeg_bytes,
            f"phase3-{run_id}.jpg",
            doctor_id,
        )
        uploaded.append((IMAGE_BUCKET, image_jpeg["file_id"]))
        check(
            image_jpeg["content_type"] == "image/jpeg"
            and download_file(IMAGE_BUCKET, image_jpeg["file_id"]) == jpeg_bytes,
            "Upload and download JPEG",
        )

        invalid_rejected = False
        try:
            upload_degree_pdf(b"not a pdf", "invalid.pdf", email)
        except ValueError:
            invalid_rejected = True
        check(invalid_rejected, "Reject non-PDF content for degree uploads")

        invalid_image_rejected = False
        try:
            upload_contribution_image(b"not an image", "invalid.png", doctor_id)
        except ValueError:
            invalid_image_rejected = True
        check(invalid_image_rejected, "Reject non-image content for image uploads")

        # Check bucket separation: each file should be found only in its bucket.
        check(
            get_file_metadata(DEGREE_BUCKET, image_png["file_id"]) is None
            and get_file_metadata(IMAGE_BUCKET, degree["file_id"]) is None,
            "Keep degree and contribution files isolated in separate buckets",
        )

        print("\n--- Cleanup ---")
        for bucket_name, file_id in reversed(uploaded):
            check(
                delete_file(bucket_name, file_id),
                f"Delete test file from {bucket_name}",
            )
        uploaded.clear()

        check(
            get_file_metadata(DEGREE_BUCKET, degree["file_id"]) is None
            and get_file_metadata(IMAGE_BUCKET, image_png["file_id"]) is None
            and get_file_metadata(IMAGE_BUCKET, image_jpeg["file_id"]) is None,
            "Verify all test files and GridFS metadata were removed",
        )

        print("\n==============================================")
        print("ALL MONGODB GRIDFS INTEGRATION TESTS PASSED")
        print("==============================================")
        return 0

    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        return 1

    finally:
        # Best-effort removal of any test files left by a failed assertion.
        for bucket_name, file_id in reversed(uploaded):
            try:
                delete_file(bucket_name, file_id)
                print(f"Cleanup: removed remaining test file from {bucket_name}.")
            except Exception as cleanup_error:
                print(
                    "WARNING: Manual MongoDB cleanup may be required: "
                    f"{type(cleanup_error).__name__}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
