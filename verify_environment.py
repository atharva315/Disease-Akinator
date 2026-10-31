"""Preflight audit for Disease Akinator's local runtime.

Run from the project root with the intended virtual environment active:
    python verify_environment.py

This script:
- reports the exact Python executable and environment;
- imports every direct runtime package used by app.py/services/database;
- prints package versions and import locations;
- compiles project Python files;
- checks required ML model files and Git LFS pointer stubs;
- checks for the presence (not contents) of local Streamlit secrets or env keys.

It never prints secret values or attempts a database connection.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path
import py_compile
import sys


ROOT = Path(__file__).resolve().parent

MODULES = {
    "streamlit": "streamlit",
    "numpy": "numpy",
    "PIL": "Pillow",
    "joblib": "joblib",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "tensorflow": "tensorflow",
    "psycopg": "psycopg",
    "pymongo": "pymongo",
    "gridfs": "pymongo",
    "bson": "pymongo",
    "argon2": "argon2-cffi",
    "dns": "dnspython",
}

MODEL_FILES = (
    "final_ensemble.pkl",
    "feature_names.pkl",
    "skin_model.h5",
    "class_names.pkl",
)

PYTHON_FILES = (
    "app.py",
    "database/__init__.py",
    "database/postgres.py",
    "database/mongodb.py",
    "services/__init__.py",
    "services/auth_service.py",
    "services/doctor_service.py",
    "services/contribution_service.py",
)

ENV_KEYS = (
    "DISEASE_AKINATOR_POSTGRES_URL",
    "DISEASE_AKINATOR_MONGODB_URI",
)


def print_header(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def check_python() -> bool:
    print_header("PYTHON ENVIRONMENT")
    print("Executable :", sys.executable)
    print("Version    :", sys.version.split()[0])
    print("Prefix     :", sys.prefix)
    print("Base prefix:", sys.base_prefix)

    is_venv = sys.prefix != sys.base_prefix
    print("Virtual env:", "YES" if is_venv else "NO")
    if not is_venv:
        print("WARNING: No virtual environment is active.")

    if sys.version_info[:2] != (3, 13):
        print("WARNING: This dependency set was prepared for Python 3.13.x.")
        return False
    print("PASS: Python 3.13 detected.")
    return True


def check_imports() -> tuple[bool, dict[str, str]]:
    print_header("RUNTIME PACKAGE IMPORTS")
    all_ok = True
    imported: dict[str, str] = {}

    for module_name, distribution_name in MODULES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            if version is None:
                try:
                    version = importlib.metadata.version(distribution_name)
                except importlib.metadata.PackageNotFoundError:
                    version = "version unavailable"

            location = getattr(module, "__file__", "built-in/namespace")
            imported[module_name] = str(location)
            print(f"PASS: {module_name:<12} {version} | {location}")
        except Exception as exc:
            all_ok = False
            print(
                f"FAIL: {module_name:<12} "
                f"{type(exc).__name__}: {exc}"
            )

    if imported.get("joblib"):
        print("\nJOBLIB DIAGNOSTIC")
        print("Imported joblib from:", imported["joblib"])
        print("This path must belong to the same .venv used to launch Streamlit.")

    return all_ok, imported


def check_python_files() -> bool:
    print_header("PYTHON SYNTAX")
    all_ok = True
    for relative in PYTHON_FILES:
        path = ROOT / relative
        if not path.is_file():
            print(f"SKIP: {relative} (not present in this folder)")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"PASS: {relative}")
        except py_compile.PyCompileError as exc:
            all_ok = False
            print(f"FAIL: {relative}: {exc}")
    return all_ok


def check_model_assets() -> bool:
    print_header("MODEL ASSETS")
    all_ok = True
    lfs_signature = b"version https://git-lfs.github.com/spec/v1"

    for relative in MODEL_FILES:
        path = ROOT / relative
        if not path.is_file():
            all_ok = False
            print(f"MISSING: {relative}")
            continue

        try:
            with path.open("rb") as handle:
                prefix = handle.read(200)
        except OSError as exc:
            all_ok = False
            print(f"FAIL: {relative}: {exc}")
            continue

        if prefix.startswith(lfs_signature):
            all_ok = False
            print(f"LFS POINTER ONLY: {relative} (real model content not downloaded)")
        else:
            print(f"PASS: {relative} ({path.stat().st_size:,} bytes)")

    if not all_ok:
        print(
            "\nWARNING: This folder is not yet a complete diagnosis runtime. "
            "Download the real Git LFS model assets into the repository root."
        )
    return all_ok


def check_configuration_presence() -> None:
    print_header("CONFIGURATION PRESENCE (VALUES NEVER DISPLAYED)")
    secrets_file = ROOT / ".streamlit" / "secrets.toml"
    print(
        "Streamlit secrets file:",
        "PRESENT" if secrets_file.is_file() else "NOT FOUND",
    )
    for name in ENV_KEYS:
        print(f"{name}:", "SET" if os.getenv(name) else "NOT SET")

    if not secrets_file.is_file() and not any(os.getenv(k) for k in ENV_KEYS):
        print(
            "WARNING: Database credentials are not configured for this local "
            "Streamlit process. The Doctor Portal database calls will fail "
            "until [postgres].url and [mongodb].uri are configured in "
            ".streamlit/secrets.toml, or the environment variables are set."
        )


def main() -> int:
    print("DISEASE AKINATOR — LOCAL ENVIRONMENT PREFLIGHT")
    python_ok = check_python()
    imports_ok, _ = check_imports()
    syntax_ok = check_python_files()
    assets_ok = check_model_assets()
    check_configuration_presence()

    print_header("SUMMARY")
    checks = {
        "Python target": python_ok,
        "Runtime imports": imports_ok,
        "Python syntax": syntax_ok,
        "Model assets": assets_ok,
    }
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL/WARNING'}: {label}")

    if all(checks.values()):
        print("\nALL PREFLIGHT CHECKS PASSED.")
        return 0

    print(
        "\nPREFLIGHT INCOMPLETE. Resolve each FAIL/WARNING above before "
        "running the diagnosis pages or considering the environment ready."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
