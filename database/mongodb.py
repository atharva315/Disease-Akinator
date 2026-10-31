"""MongoDB Atlas GridFS utilities for Disease Akinator.

Binary data is stored in two dedicated GridFS buckets:
- doctor_degrees
- contribution_images

The PostgreSQL database stores structured records and the returned GridFS IDs.
This module never stores credentials in source code or logs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from bson import ObjectId
from gridfs import GridFSBucket
from gridfs.errors import NoFile
from pymongo import MongoClient


DATABASE_NAME = "disease_akinator"
DEGREE_BUCKET = "doctor_degrees"
IMAGE_BUCKET = "contribution_images"

MAX_DEGREE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 5 * 1024 * 1024

_PDF_SIGNATURE = b"%PDF-"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"


class MongoConfigurationError(RuntimeError):
    """Raised when MongoDB connection settings are unavailable."""


def _get_uri() -> str:
    """Read the Atlas URI from an environment variable or Streamlit secrets."""
    env_uri = os.getenv("DISEASE_AKINATOR_MONGODB_URI")
    if env_uri and env_uri.strip():
        return env_uri.strip()

    try:
        import streamlit as st

        uri = st.secrets["mongodb"]["uri"]
    except Exception as exc:
        raise MongoConfigurationError(
            "MongoDB settings are missing. Set "
            "DISEASE_AKINATOR_MONGODB_URI or configure "
            "[mongodb].uri in Streamlit secrets."
        ) from exc

    if not isinstance(uri, str) or not uri.strip():
        raise MongoConfigurationError(
            "Streamlit secret mongodb.uri must be a non-empty string."
        )
    return uri.strip()


@lru_cache(maxsize=4)
def _cached_client(uri: str) -> MongoClient:
    """Reuse a thread-safe MongoClient and its connection pool per URI."""
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=10_000,
        connectTimeoutMS=10_000,
        socketTimeoutMS=20_000,
        appname="DiseaseAkinator",
    )


def get_mongo_client(uri: str | None = None) -> MongoClient:
    """Return a pooled MongoClient; callers should not close it per request."""
    connection_uri = uri.strip() if isinstance(uri, str) else _get_uri()
    if not connection_uri:
        raise MongoConfigurationError("MongoDB connection URI is empty.")

    parsed = urlsplit(connection_uri)
    if parsed.scheme not in {"mongodb", "mongodb+srv"} or not parsed.hostname:
        raise MongoConfigurationError("MongoDB URI must use mongodb:// or mongodb+srv://.")

    return _cached_client(connection_uri)


def get_mongo_database(uri: str | None = None):
    """Return the configured Disease Akinator database."""
    return get_mongo_client(uri)[DATABASE_NAME]


def check_mongodb_connection(uri: str | None = None) -> dict[str, Any]:
    """Ping Atlas and return safe connection identity information."""
    client = get_mongo_client(uri)
    result = client.admin.command("ping")
    return {
        "connected": result.get("ok") == 1.0,
        "database": DATABASE_NAME,
        "server": "MongoDB Atlas",
    }


def _require_bytes(file_bytes: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(file_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("file_bytes must be bytes-like data.")
    data = bytes(file_bytes)
    if not data:
        raise ValueError("Uploaded file is empty.")
    return data


def _normalise_email(email: str) -> str:
    if not isinstance(email, str) or not email.strip():
        raise ValueError("Uploader email is required.")
    return email.strip().lower()


def _validate_pdf(
    file_bytes: bytes | bytearray | memoryview,
    filename: str,
) -> bytes:
    data = _require_bytes(file_bytes)
    if not isinstance(filename, str) or Path(filename).suffix.lower() != ".pdf":
        raise ValueError("Doctor degree must have a .pdf filename.")
    if len(data) > MAX_DEGREE_BYTES:
        raise ValueError("Doctor degree PDF exceeds the 10 MB limit.")
    if not data.startswith(_PDF_SIGNATURE):
        raise ValueError("Doctor degree file does not have a PDF signature.")
    return data


def _validate_image(
    file_bytes: bytes | bytearray | memoryview,
    filename: str,
) -> tuple[bytes, str]:
    data = _require_bytes(file_bytes)
    if not isinstance(filename, str):
        raise ValueError("Image filename is required.")

    suffix = Path(filename).suffix.lower()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Contribution image exceeds the 5 MB limit.")

    if data.startswith(_PNG_SIGNATURE):
        content_type = "image/png"
        if suffix != ".png":
            raise ValueError("PNG image must use a .png filename.")
    elif data.startswith(_JPEG_SIGNATURE):
        content_type = "image/jpeg"
        if suffix not in {".jpg", ".jpeg"}:
            raise ValueError("JPEG image must use a .jpg or .jpeg filename.")
    else:
        raise ValueError("Only valid PNG and JPEG image formats are accepted.")

    return data, content_type


def _get_bucket(bucket_name: str) -> GridFSBucket:
    if bucket_name not in {DEGREE_BUCKET, IMAGE_BUCKET}:
        raise ValueError("Unsupported GridFS bucket.")
    return GridFSBucket(get_mongo_database(), bucket_name=bucket_name)


def upload_degree_pdf(
    file_bytes: bytes | bytearray | memoryview,
    filename: str,
    uploaded_by_email: str,
) -> dict[str, Any]:
    """Upload a PDF degree file and return its GridFS ID and metadata."""
    data = _validate_pdf(file_bytes, filename)
    email = _normalise_email(uploaded_by_email)
    bucket = _get_bucket(DEGREE_BUCKET)

    metadata = {
        "file_type": "doctor_degree",
        "uploaded_by_email": email,
        "uploaded_at": datetime.now(timezone.utc),
        "content_type": "application/pdf",
        "original_filename": Path(filename).name,
        "file_size_bytes": len(data),
    }
    file_id = bucket.upload_from_stream(
        Path(filename).name,
        data,
        metadata=metadata,
    )
    return {
        "file_id": str(file_id),
        "bucket": DEGREE_BUCKET,
        "filename": Path(filename).name,
        "content_type": "application/pdf",
        "file_size_bytes": len(data),
    }


def upload_contribution_image(
    file_bytes: bytes | bytearray | memoryview,
    filename: str,
    uploaded_by_doctor_id: int,
) -> dict[str, Any]:
    """Upload a PNG/JPEG contribution image and return its GridFS ID."""
    if isinstance(uploaded_by_doctor_id, bool) or not isinstance(
        uploaded_by_doctor_id, int
    ) or uploaded_by_doctor_id <= 0:
        raise ValueError("A positive uploader doctor ID is required.")

    data, content_type = _validate_image(file_bytes, filename)
    bucket = _get_bucket(IMAGE_BUCKET)

    metadata = {
        "file_type": "contribution_image",
        "uploaded_by_doctor_id": uploaded_by_doctor_id,
        "uploaded_at": datetime.now(timezone.utc),
        "content_type": content_type,
        "original_filename": Path(filename).name,
        "file_size_bytes": len(data),
    }
    file_id = bucket.upload_from_stream(
        Path(filename).name,
        data,
        metadata=metadata,
    )
    return {
        "file_id": str(file_id),
        "bucket": IMAGE_BUCKET,
        "filename": Path(filename).name,
        "content_type": content_type,
        "file_size_bytes": len(data),
    }


def _parse_file_id(file_id: str) -> ObjectId:
    if not isinstance(file_id, str) or not ObjectId.is_valid(file_id):
        raise ValueError("Invalid MongoDB GridFS file ID.")
    return ObjectId(file_id)


def download_file(bucket_name: str, file_id: str) -> bytes:
    """Download and return the original binary file from its GridFS bucket."""
    bucket = _get_bucket(bucket_name)
    try:
        with bucket.open_download_stream(_parse_file_id(file_id)) as stream:
            return stream.read()
    except NoFile as exc:
        raise FileNotFoundError("GridFS file was not found.") from exc


def get_file_metadata(bucket_name: str, file_id: str) -> dict[str, Any] | None:
    """Return safe file metadata, converting ObjectId to a string."""
    if bucket_name not in {DEGREE_BUCKET, IMAGE_BUCKET}:
        raise ValueError("Unsupported GridFS bucket.")

    record = get_mongo_database()[f"{bucket_name}.files"].find_one(
        {"_id": _parse_file_id(file_id)}
    )
    if record is None:
        return None

    return {
        "file_id": str(record["_id"]),
        "filename": record.get("filename"),
        "length": record.get("length"),
        "upload_date": record.get("uploadDate"),
        "metadata": record.get("metadata", {}),
    }


def delete_file(bucket_name: str, file_id: str) -> bool:
    """Delete a file and its chunks from the selected GridFS bucket."""
    bucket = _get_bucket(bucket_name)
    object_id = _parse_file_id(file_id)
    files_collection = get_mongo_database()[f"{bucket_name}.files"]

    if files_collection.find_one({"_id": object_id}, {"_id": 1}) is None:
        return False

    bucket.delete(object_id)
    return True


def close_cached_clients() -> None:
    """Close cached Mongo clients; intended for tests and process shutdown."""
    for client in _cached_client.cache_info() and []:
        client.close()
    _cached_client.cache_clear()
