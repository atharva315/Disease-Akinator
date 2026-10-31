"""PostgreSQL access layer for Disease Akinator.

Uses short-lived Psycopg connections. The Streamlit application supplies
the DSN through st.secrets["postgres"]["url"]; tests may pass a DSN directly.
Schema changes are intentionally kept out of this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg
from psycopg.rows import dict_row


class DatabaseConfigurationError(RuntimeError):
    """Raised when PostgreSQL connection settings are unavailable."""


def _get_secret_conninfo() -> str:
    """Read the PostgreSQL URI from Streamlit secrets without logging it."""
    try:
        import streamlit as st

        conninfo = st.secrets["postgres"]["url"]
    except Exception as exc:
        raise DatabaseConfigurationError(
            "PostgreSQL connection settings are missing. "
            "Configure [postgres].url in Streamlit secrets."
        ) from exc

    if not isinstance(conninfo, str) or not conninfo.strip():
        raise DatabaseConfigurationError(
            "Streamlit secret postgres.url must be a non-empty string."
        )
    return conninfo.strip()


def get_connection(conninfo: str | None = None) -> psycopg.Connection:
    """Create a PostgreSQL connection.

    Callers should use it as a context manager:
        with get_connection() as conn:
            ...

    Passing conninfo explicitly makes this layer straightforward to test.
    """
    dsn = conninfo.strip() if isinstance(conninfo, str) else _get_secret_conninfo()
    if not dsn:
        raise DatabaseConfigurationError("PostgreSQL connection URI is empty.")

    return psycopg.connect(
        dsn,
        connect_timeout=15,
        row_factory=dict_row,
    )


def check_database_connection(conninfo: str | None = None) -> dict[str, str]:
    """Return the authenticated PostgreSQL identity for health checks."""
    with get_connection(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT current_user AS database_user, "
                "current_database() AS database_name"
            )
            row = cur.fetchone()
    return row


def _normalise_email(email: str) -> str:
    if not isinstance(email, str) or not email.strip():
        raise ValueError("Email is required.")
    return email.strip().lower()


def _clean_symptoms(symptoms: Sequence[str]) -> list[str]:
    if isinstance(symptoms, (str, bytes)):
        raise TypeError("Symptoms must be a sequence of individual strings.")

    cleaned: list[str] = []
    seen: set[str] = set()
    for symptom in symptoms:
        if not isinstance(symptom, str):
            raise TypeError("Every symptom must be a string.")
        value = symptom.strip()
        key = value.casefold()
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)

    if len(cleaned) < 2:
        raise ValueError("At least two distinct symptoms are required.")
    if len(cleaned) > 32767:
        raise ValueError("Too many symptoms in one contribution.")
    return cleaned


# ---------------------------------------------------------------------
# DOCTORS
# ---------------------------------------------------------------------

def create_doctor(
    *,
    full_name: str,
    age: int,
    degree: str,
    custom_degree: str | None,
    hospital: str | None,
    city: str | None,
    email: str,
    password_hash: str,
    degree_file_id: str,
    degree_filename: str,
    degree_size_bytes: int,
) -> dict[str, Any]:
    """Insert a doctor profile and return its non-secret fields.

    The degree PDF is uploaded by the file service in Phase 3; this function
    stores the returned GridFS file ID and metadata in PostgreSQL.
    """
    normalized_email = _normalise_email(email)
    if not password_hash or not password_hash.strip():
        raise ValueError("Password hash is required.")

    query = """
        INSERT INTO public.doctors (
            full_name, age, degree, custom_degree, hospital, city,
            email, password_hash, degree_file_id, degree_filename,
            degree_size_bytes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING
            id, full_name, age, degree, custom_degree, hospital, city,
            email, degree_file_id, degree_filename, degree_size_bytes,
            created_at
    """
    values = (
        full_name.strip(),
        age,
        degree,
        custom_degree.strip() if custom_degree else None,
        hospital.strip() if hospital and hospital.strip() else None,
        city.strip() if city and city.strip() else None,
        normalized_email,
        password_hash,
        degree_file_id,
        degree_filename.strip(),
        degree_size_bytes,
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)
            return cur.fetchone()


def get_doctor_by_email(email: str) -> dict[str, Any] | None:
    """Fetch a doctor's profile without returning the password hash."""
    query = """
        SELECT
            id, full_name, age, degree, custom_degree, hospital, city,
            email, degree_file_id, degree_filename, degree_size_bytes,
            created_at
        FROM public.doctors
        WHERE email = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (_normalise_email(email),))
            return cur.fetchone()


def get_doctor_auth_record(email: str) -> dict[str, Any] | None:
    """Fetch only the fields required by the future authentication service."""
    query = """
        SELECT id, full_name, email, degree, password_hash
        FROM public.doctors
        WHERE email = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (_normalise_email(email),))
            return cur.fetchone()


def get_doctor_by_id(doctor_id: int) -> dict[str, Any] | None:
    query = """
        SELECT
            id, full_name, age, degree, custom_degree, hospital, city,
            email, degree_file_id, degree_filename, degree_size_bytes,
            created_at
        FROM public.doctors
        WHERE id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (doctor_id,))
            return cur.fetchone()


def update_doctor_profile(
    doctor_id: int,
    *,
    full_name: str,
    age: int,
    degree: str,
    custom_degree: str | None,
    hospital: str | None,
    city: str | None,
) -> dict[str, Any] | None:
    query = """
        UPDATE public.doctors
        SET full_name = %s,
            age = %s,
            degree = %s,
            custom_degree = %s,
            hospital = %s,
            city = %s
        WHERE id = %s
        RETURNING id, full_name, age, degree, custom_degree,
                  hospital, city, email, created_at
    """
    values = (
        full_name.strip(),
        age,
        degree,
        custom_degree.strip() if custom_degree else None,
        hospital.strip() if hospital and hospital.strip() else None,
        city.strip() if city and city.strip() else None,
        doctor_id,
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)
            return cur.fetchone()


def delete_doctor(doctor_id: int) -> bool:
    """Delete a doctor only when FK restrictions allow it."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.doctors WHERE id = %s RETURNING id",
                (doctor_id,),
            )
            return cur.fetchone() is not None


# ---------------------------------------------------------------------
# SYMPTOM CONTRIBUTIONS
# ---------------------------------------------------------------------

def create_symptom_contribution(
    doctor_id: int,
    disease_name: str,
    symptoms: Sequence[str],
    notes: str | None = None,
) -> dict[str, Any]:
    """Create one contribution and its ordered symptom rows atomically."""
    cleaned = _clean_symptoms(symptoms)
    disease = disease_name.strip() if isinstance(disease_name, str) else ""
    if not disease:
        raise ValueError("Disease name is required.")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.symptom_contributions
                    (doctor_id, disease_name, notes)
                VALUES (%s, %s, %s)
                RETURNING id, doctor_id, disease_name, notes, submitted_at
                """,
                (doctor_id, disease, notes),
            )
            contribution = cur.fetchone()

            cur.executemany(
                """
                INSERT INTO public.symptom_contribution_items
                    (contribution_id, position, symptom_text)
                VALUES (%s, %s, %s)
                """,
                [
                    (contribution["id"], position, symptom)
                    for position, symptom in enumerate(cleaned, start=1)
                ],
            )

    contribution["symptoms"] = cleaned
    return contribution


def get_symptom_contribution(
    contribution_id: int,
) -> dict[str, Any] | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, doctor_id, disease_name, notes, submitted_at
                FROM public.symptom_contributions
                WHERE id = %s
                """,
                (contribution_id,),
            )
            contribution = cur.fetchone()
            if contribution is None:
                return None

            cur.execute(
                """
                SELECT symptom_text
                FROM public.symptom_contribution_items
                WHERE contribution_id = %s
                ORDER BY position
                """,
                (contribution_id,),
            )
            contribution["symptoms"] = [
                row["symptom_text"] for row in cur.fetchall()
            ]
            return contribution


def list_symptom_contributions(
    doctor_id: int | None = None,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if not 1 <= limit <= 500 or offset < 0:
        raise ValueError("Use limit 1–500 and a non-negative offset.")

    query = """
        SELECT
            sc.id, sc.doctor_id, sc.disease_name, sc.notes, sc.submitted_at,
            d.full_name AS uploaded_by,
            COALESCE(
                ARRAY_AGG(sci.symptom_text ORDER BY sci.position)
                    FILTER (WHERE sci.symptom_text IS NOT NULL),
                ARRAY[]::VARCHAR[]
            ) AS symptoms
        FROM public.symptom_contributions sc
        JOIN public.doctors d ON d.id = sc.doctor_id
        LEFT JOIN public.symptom_contribution_items sci
            ON sci.contribution_id = sc.id
        WHERE (%s IS NULL OR sc.doctor_id = %s)
        GROUP BY sc.id, d.full_name
        ORDER BY sc.submitted_at DESC, sc.id DESC
        LIMIT %s OFFSET %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (doctor_id, doctor_id, limit, offset))
            return cur.fetchall()


def update_symptom_contribution(
    contribution_id: int,
    *,
    disease_name: str,
    symptoms: Sequence[str],
    notes: str | None = None,
) -> dict[str, Any] | None:
    cleaned = _clean_symptoms(symptoms)
    disease = disease_name.strip() if isinstance(disease_name, str) else ""
    if not disease:
        raise ValueError("Disease name is required.")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.symptom_contributions
                SET disease_name = %s, notes = %s
                WHERE id = %s
                RETURNING id, doctor_id, disease_name, notes, submitted_at
                """,
                (disease, notes, contribution_id),
            )
            contribution = cur.fetchone()
            if contribution is None:
                return None

            cur.execute(
                """
                DELETE FROM public.symptom_contribution_items
                WHERE contribution_id = %s
                """,
                (contribution_id,),
            )
            cur.executemany(
                """
                INSERT INTO public.symptom_contribution_items
                    (contribution_id, position, symptom_text)
                VALUES (%s, %s, %s)
                """,
                [
                    (contribution_id, position, symptom)
                    for position, symptom in enumerate(cleaned, start=1)
                ],
            )

    contribution["symptoms"] = cleaned
    return contribution


def delete_symptom_contribution(contribution_id: int) -> bool:
    """Delete parent contribution; symptom rows cascade automatically."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM public.symptom_contributions
                WHERE id = %s
                RETURNING id
                """,
                (contribution_id,),
            )
            return cur.fetchone() is not None


# ---------------------------------------------------------------------
# IMAGE CONTRIBUTIONS (metadata only; binary files remain in GridFS)
# ---------------------------------------------------------------------

def create_image_contribution(
    doctor_id: int,
    disease_name: str,
    mongo_file_id: str,
    image_filename: str,
    content_type: str,
    file_size_bytes: int,
    notes: str | None = None,
) -> dict[str, Any]:
    query = """
        INSERT INTO public.image_contributions (
            doctor_id, disease_name, mongo_file_id, image_filename,
            content_type, file_size_bytes, notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id, doctor_id, disease_name, mongo_file_id,
                  image_filename, content_type, file_size_bytes,
                  notes, submitted_at
    """
    values = (
        doctor_id,
        disease_name.strip(),
        mongo_file_id,
        image_filename.strip(),
        content_type,
        file_size_bytes,
        notes,
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)
            return cur.fetchone()


def get_image_contribution(
    contribution_id: int,
) -> dict[str, Any] | None:
    query = """
        SELECT
            ic.id, ic.doctor_id, ic.disease_name, ic.mongo_file_id,
            ic.image_filename, ic.content_type, ic.file_size_bytes,
            ic.notes, ic.submitted_at, d.full_name AS uploaded_by
        FROM public.image_contributions ic
        JOIN public.doctors d ON d.id = ic.doctor_id
        WHERE ic.id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (contribution_id,))
            return cur.fetchone()


def list_image_contributions(
    doctor_id: int | None = None,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if not 1 <= limit <= 500 or offset < 0:
        raise ValueError("Use limit 1–500 and a non-negative offset.")

    query = """
        SELECT
            ic.id, ic.doctor_id, ic.disease_name, ic.mongo_file_id,
            ic.image_filename, ic.content_type, ic.file_size_bytes,
            ic.notes, ic.submitted_at, d.full_name AS uploaded_by
        FROM public.image_contributions ic
        JOIN public.doctors d ON d.id = ic.doctor_id
        WHERE (%s IS NULL OR ic.doctor_id = %s)
        ORDER BY ic.submitted_at DESC, ic.id DESC
        LIMIT %s OFFSET %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (doctor_id, doctor_id, limit, offset))
            return cur.fetchall()


def update_image_contribution(
    contribution_id: int,
    *,
    disease_name: str,
    notes: str | None = None,
) -> dict[str, Any] | None:
    query = """
        UPDATE public.image_contributions
        SET disease_name = %s, notes = %s
        WHERE id = %s
        RETURNING id, doctor_id, disease_name, mongo_file_id,
                  image_filename, content_type, file_size_bytes,
                  notes, submitted_at
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (disease_name.strip(), notes, contribution_id),
            )
            return cur.fetchone()


def delete_image_contribution(contribution_id: int) -> bool:
    """Delete PostgreSQL metadata only. GridFS deletion belongs to Phase 3."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM public.image_contributions
                WHERE id = %s
                RETURNING id
                """,
                (contribution_id,),
            )
            return cur.fetchone() is not None


# ---------------------------------------------------------------------
# CONTRIBUTION STATISTICS
# ---------------------------------------------------------------------

def get_contribution_stats() -> dict[str, int]:
    query = """
        SELECT
            (SELECT COUNT(*) FROM public.doctors) AS doctors,
            (SELECT COUNT(*) FROM public.symptom_contributions)
                AS symptom_entries,
            (SELECT COUNT(*) FROM public.image_contributions)
                AS image_entries
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
            return {key: int(value) for key, value in row.items()}
