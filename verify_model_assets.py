"""Load and smoke-test every Disease Akinator Git LFS model artifact.

Run from the repository root with the project .venv active:
    python verify_model_assets.py

Loads project-owned pickle/Keras artifacts and performs one synthetic zero-input
inference for each prediction model; no real patient data is used.
The symptom model input width and image input shape follow app.py:
- symptom model: one row with len(feature_names) features
- skin model: one RGB image resized to 160x160 in [0, 1]

The imputer artifact is loaded and inspected, but app.py currently does not
use it in its live symptom prediction path.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
import sys

# Match the application CPU configuration before importing TensorFlow.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import sklearn
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "final_ensemble.pkl",
    "feature_names.pkl",
    "imputer.pkl",
    "skin_model.h5",
    "class_names.pkl",
)
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def require_artifact(name: str) -> Path:
    path = ROOT / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    with path.open("rb") as handle:
        if handle.read(200).startswith(LFS_POINTER):
            raise RuntimeError(
                f"{name} is still a Git LFS pointer. Run git lfs pull."
            )
    print(f"PASS: Found {name} ({path.stat().st_size:,} bytes)")
    return path


def main() -> int:
    print("=== Disease Akinator Model Asset Compatibility Test ===")
    print("Python       :", sys.version.split()[0])
    print("NumPy        :", np.__version__)
    print("Scikit-learn :", sklearn.__version__)
    print("Joblib       :", joblib.__version__)
    print("TensorFlow   :", tf.__version__)

    paths = {name: require_artifact(name) for name in REQUIRED}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        symptom_model = joblib.load(paths["final_ensemble.pkl"])
        feature_names = joblib.load(paths["feature_names.pkl"])
        imputer = joblib.load(paths["imputer.pkl"])
        class_names = joblib.load(paths["class_names.pkl"])

    for warning in caught:
        print("WARNING:", warning.category.__name__, str(warning.message))

    print("\n--- Symptom model ---")
    if not hasattr(feature_names, "__len__") or len(feature_names) == 0:
        raise TypeError("feature_names.pkl must contain a non-empty sequence.")
    feature_count = len(feature_names)
    print("PASS: Feature names loaded:", feature_count)
    print("PASS: Symptom model type:", type(symptom_model).__name__)

    if not callable(getattr(symptom_model, "predict_proba", None)):
        raise TypeError("Symptom model does not provide predict_proba().")
    if not hasattr(symptom_model, "classes_"):
        raise TypeError("Symptom model does not expose classes_.")

    model_features = getattr(symptom_model, "n_features_in_", None)
    if model_features is not None and int(model_features) != feature_count:
        raise ValueError(
            f"Symptom model expects {model_features} features, but "
            f"feature_names.pkl contains {feature_count}."
        )
    print("PASS: Symptom model feature width matches feature_names.pkl")
    print("Symptom classes:", len(symptom_model.classes_))

    symptom_sample = np.zeros((1, feature_count), dtype=np.float64)
    symptom_probabilities = np.asarray(
        symptom_model.predict_proba(symptom_sample)
    )
    if (
        symptom_probabilities.ndim != 2
        or symptom_probabilities.shape[0] != 1
        or symptom_probabilities.shape[1] != len(symptom_model.classes_)
        or not np.isfinite(symptom_probabilities).all()
    ):
        raise ValueError(
            "Symptom model returned an invalid probability array: "
            f"{symptom_probabilities.shape}"
        )
    print("PASS: Symptom model synthetic inference:", symptom_probabilities.shape)

    print("\n--- Imputer artifact ---")
    print("PASS: Imputer loaded:", type(imputer).__name__)
    if callable(getattr(imputer, "transform", None)):
        imputer_features = getattr(imputer, "n_features_in_", None)
        if imputer_features is not None:
            print("Imputer fitted feature width:", int(imputer_features))
            if int(imputer_features) != feature_count:
                print(
                    "NOTE: Imputer width differs from the symptom feature "
                    "list. app.py does not currently use imputer.pkl."
                )
        print("PASS: Imputer exposes transform()")
    else:
        print(
            "NOTE: imputer.pkl does not expose transform(); it is currently "
            "not used by app.py."
        )

    print("\n--- Skin image model ---")
    skin_model = tf.keras.models.load_model(str(paths["skin_model.h5"]), compile=False)
    print("PASS: Skin model loaded:", type(skin_model).__name__)
    print("Skin model input shape:", skin_model.input_shape)
    print("Skin model output shape:", skin_model.output_shape)

    input_shape = skin_model.input_shape
    if isinstance(input_shape, list):
        raise ValueError(
            "Skin model has multiple inputs, but app.py supplies one RGB image."
        )
    if tuple(input_shape[1:]) != (160, 160, 3):
        raise ValueError(
            "Skin model input shape does not match app.py preprocessing "
            f"(expected (None, 160, 160, 3), got {input_shape})."
        )

    if not hasattr(class_names, "__len__") or len(class_names) == 0:
        raise TypeError("class_names.pkl must contain a non-empty sequence.")

    skin_sample = np.zeros((1, 160, 160, 3), dtype=np.float32)
    skin_output = np.asarray(skin_model(skin_sample, training=False))
    if skin_output.ndim != 2 or skin_output.shape[0] != 1:
        raise ValueError(
            f"Skin model returned an unexpected output shape: {skin_output.shape}"
        )
    if skin_output.shape[1] != len(class_names):
        raise ValueError(
            f"Skin model produces {skin_output.shape[1]} outputs, but "
            f"class_names.pkl contains {len(class_names)} labels."
        )
    if not np.isfinite(skin_output).all():
        raise ValueError("Skin model returned non-finite outputs.")

    print("PASS: Skin model input matches 160x160 RGB preprocessing")
    print("PASS: Skin output count matches class_names.pkl:", len(class_names))
    print("PASS: Skin model synthetic inference:", skin_output.shape)

    print("\n==============================================")
    print("ALL MODEL ASSET COMPATIBILITY TESTS PASSED")
    print("==============================================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
