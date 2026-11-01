"""Manual MongoDB Atlas/GridFS integration test for Disease Akinator.

Run from the repository root:
    python tests/test_mongodb_integration.py

Credentials are entered through hidden prompts and are not written to files.
Uploads uniquely generated minimal PDF/PNG/JPEG fixtures, downloads and
compares bytes, checks uploader metadata, rejects invalid formats, then
deletes every test file.
"""

from __future__ import annotations

import getpass
import os
import sys
import base64
from pathlib import Path
from urllib.parse import urlsplit, unquote
from uuid import uuid4



PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_test_pdf() -> bytes:
    """Generate a valid one-page PDF fixture without external tools."""
    text = b"BT /F1 12 Tf 20 60 Td (Disease Akinator GridFS test) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(text)).encode()
        + b" >>\nstream\n" + text + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def make_test_png() -> bytes:
    """Return a valid 1x1 RGB PNG fixture."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
        "AAAADElEQVR4nGO4o6EBAAMQAS0ujiXaAAAAAElFTkSuQmCC"
    )


def make_test_jpeg() -> bytes:
    """Return a valid 2x2 JPEG fixture."""
    return base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/"
        "2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAACAAIDASIAAhEBAxEB/"
        "8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/"
        "8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/"
        "9oADAMBAAIRAxEAPwDiaKKK+PP6QP/Z"
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

    # Valid synthetic fixtures only; this does not test medical authenticity.
    pdf_bytes = make_test_pdf()
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
