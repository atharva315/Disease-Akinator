# Disease Akinator — Dataset, Model, and Production Lifecycle

This document records the end-to-end lifecycle that can be verified from the current public Disease Akinator repository and deployed application. It deliberately separates **confirmed implementation details** from **training provenance that is not currently archived in this repository**.

- Production repository: https://github.com/atharva315/Disease-Akinator
- Production application: https://disease-akinator-we7wpcyvg2tc4niefueebu.streamlit.app/
- Default branch: `main`

## 1. Product scope

Disease Akinator is a Streamlit educational prototype with three user-facing areas:

1. Symptom-based disease prediction.
2. Skin-image classification.
3. Doctor Portal for account registration, login, and labelled symptom/image contributions.

The deployed application is an inference and contribution application. It does not train either prediction model at startup and does not automatically retrain models from submitted contributions.

> **Medical limitation:** This is not a clinically validated medical device. Predictions and confidence scores are experimental model outputs, not diagnoses, calibrated clinical certainty, or treatment advice. Users should consult qualified healthcare professionals.

## 2. Dataset provenance — what is and is not recorded

The public repository contains the deployed model artifacts and their runtime label/feature files. It does **not** contain the original raw training datasets or a complete dataset-audit package.

| Workstream | What the repository confirms | What is not recorded in this repository |
|---|---|---|
| Symptom-to-disease | The deployed model consumes the ordered feature list in `feature_names.pkl` and returns class probabilities through `predict_proba`. | Exact raw dataset name/URL/revision, licence, row count, disease counts, symptom counts, cleaning rules, duplicate/conflict audit, and source attribution. |
| Skin-image classification | The image screen labels the model as **ResNet50 (DermNet NZ)**. The deployed class mapping is loaded from `class_names.pkl`. | Exact dataset package/version or download URL, licence/permission record, class-folder inventory, total/per-class image counts, corrupt/duplicate-image report, patient/case grouping, and split manifest. |

The statement “ResNet50 (DermNet NZ)” is the model/source label displayed by the application. The repository does not include enough of the original training record to independently establish the exact dataset release, licence, or training split. Confirm those details from the original training workspace before presenting them as reproducible facts.

### Why the raw datasets are not committed

The repository should contain code, metadata, and evaluation reports—not private, restricted, or non-redistributable medical images or raw data. A future reproducibility package should record the source URL, version/revision, access date, licence/terms, attribution, dataset checksum, and class/row counts without copying restricted source data into GitHub.

## 3. Model artifacts delivered to production

The following artifacts are tracked with Git LFS:

| Artifact | Runtime purpose |
|---|---|
| `final_ensemble.pkl` | Serialized symptom prediction classifier/ensemble |
| `feature_names.pkl` | Ordered symptom feature schema used to construct model inputs |
| `imputer.pkl` | Retained preprocessing artifact; the current live inference path does not call it |
| `skin_model.h5` | TensorFlow/Keras skin-image classifier |
| `class_names.pkl` | Output-index-to-skin-class mapping |

The public Git tree contains Git LFS pointer files for the large model artifacts. A clone therefore needs Git LFS installed and `git lfs pull` before model loading.

The model artifacts are deployment outputs. They do not, by themselves, preserve the original training dataset, training code, hyperparameters, split methodology, or evaluation evidence.

## 4. Current inference pipelines

These details are taken from the production `app.py` implementation.

### 4.1 Symptom prediction

1. Load `final_ensemble.pkl` and `feature_names.pkl` with Joblib.
2. Create a zero-initialized input row with one position for every feature in the saved feature list.
3. Ask the next unasked feature in the saved feature-list order. The current application does not perform entropy-based or learned question ranking.
4. Map each response to the numeric value expected by the current application:
   - Yes → `1.0`
   - Probably → `0.8`
   - Probably Not → `0.2`
   - No → `0.0`
5. Call the classifier's `predict_proba` after each answer.
6. Finish a question round when the maximum class probability reaches the configured `0.87` threshold or when 25 questions have been asked in that round. At the checkpoint, the user can accept the best guess or continue.
7. Display the highest-probability class and up to three alternatives.

The deployed application uses the saved feature order as its inference schema. A different feature order, different value encoding, or a mismatched model artifact would invalidate predictions.

### 4.2 Skin-image classification

1. Accept a JPG/JPEG or PNG upload.
2. Decode the image with Pillow.
3. Convert it to RGB.
4. Resize it to `160 × 160` pixels.
5. Convert the pixel array to `float32` and divide by `255.0`, producing values in the `[0, 1]` range.
6. Add a batch dimension and call the Keras model in inference mode using `skin_model(image, training=False)`.
7. Map output indices through `class_names.pkl` and display the top five scores.

The production app loads the image model lazily, uses `compile=False`, and installs CPU-only TensorFlow `2.20.0` for the CPU-hosted Streamlit runtime. These are deployment/runtime choices, not claims about the original model-training environment.

## 5. Training and evaluation record — reproducibility gap

The current repository does not include the training notebooks/scripts or the complete evidence needed to recreate the deployed models from raw datasets. The following items are therefore **not independently reproducible from this repository today**:

- Dataset source, exact version/revision, licence and attribution for both model pipelines.
- Dataset audit reports, class/label distributions, duplicate/corruption checks, and exclusion rules.
- Train/validation/test split sizes, random seeds, and leakage controls.
- Dataset-level preprocessing and augmentation code.
- Candidate models, training configuration, hyperparameters, early-stopping/checkpoint rules, and experiment logs.
- Final evaluation reports, confusion matrices, per-class precision/recall, macro-F1, and held-out test protocol.

The README reports project-level figures of approximately **91% symptom-prediction accuracy** and **82% skin-image classification accuracy**. These are historical project-reported figures. A corresponding evaluation report and test protocol are not included in this repository, so readers cannot independently reproduce or audit those values from the public files alone. They must not be represented as clinical performance.

### Artifacts to add if the original training workspace is available

To close this reproducibility gap in a future documentation update, export a small, non-sensitive record for each model:

- `dataset_manifest.md`: dataset name, source URL, version/revision, licence/terms, attribution, access date, checksum, and dataset size.
- `dataset_audit.md`: schema/classes, class distribution, missing/corrupt records, duplicate/conflicting-label checks, exclusions, and known limitations.
- `split_manifest.csv` or a split summary: train/validation/test counts, random seed, grouping strategy, and leakage checks. Do not publish patient-identifying metadata.
- `preprocessing.md` and the versioned preprocessing code: exact transforms and feature mapping.
- `training_config.yaml`: architecture, parameters, hyperparameters, seed, environment/library versions, and checkpoint/selection criteria.
- `evaluation_report.md`: held-out test metrics, confusion matrix, per-class precision/recall, macro-F1, test methodology, and limitations.

Do not invent these fields from the deployed model files. If the original workspace no longer exists, record the missing provenance as unavailable rather than reverse-engineering or guessing it.

## 6. Doctor Portal and data storage

The Doctor Portal is a contribution and storage workflow. It does not approve or verify professional credentials and does not automatically feed new contributions into model training.

### Neon PostgreSQL

The schema in `database/schema.sql` defines:

- `doctors`: profile fields, normalized email, password hash, degree-file reference and upload metadata.
- `symptom_contributions`: doctor-linked disease and contribution records.
- `symptom_contribution_items`: individual symptom entries linked to a symptom contribution.
- `image_contributions`: doctor-linked disease, image metadata and GridFS file reference.

### MongoDB Atlas GridFS

The application stores binary files in two GridFS buckets:

- `doctor_degrees`: uploaded degree PDFs.
- `contribution_images`: uploaded labelled images.

The PostgreSQL records retain file IDs and metadata; the binary content is stored in GridFS. The service layer includes cleanup/compensating-delete logic for failures between file upload and relational metadata insertion.

### Authentication and uploads

- Doctor passwords are hashed with Argon2id; plaintext passwords are not stored.
- Registration requires a password of at least 12 characters.
- Degree upload: PDF, maximum 10 MB.
- Contribution image upload: JPEG/PNG, maximum 5 MB.
- The uploaded degree document is not independently validated by a credential authority.
- Symptom and image contributions are stored for future review; they do not trigger model retraining.

## 7. Production architecture and deployment

The deployed system is a single Streamlit application with local model inference and database-backed Doctor Portal operations:

```text
Browser
  |
  v
Streamlit Community Cloud (app.py, main branch)
  |                         |
  |                         +--> Joblib symptom model + feature schema
  |                         |
  |                         +--> TensorFlow/Keras image model + class map
  |
  +--> Neon PostgreSQL (profiles, contributions, file metadata)
  |
  +--> MongoDB Atlas GridFS (degree PDFs, contribution images)
```

### Configuration

The application reads database connection values from Streamlit secrets or the following environment variables:

- `DISEASE_AKINATOR_POSTGRES_URL`
- `DISEASE_AKINATOR_MONGODB_URI`

Local development uses an ignored `.streamlit/secrets.toml`. Production credentials are configured in the Streamlit Cloud app settings. Secrets must never be committed to GitHub or pasted into issues, screenshots, or documentation.

### Deployment sequence used

1. Keep changes isolated on a feature/release branch.
2. Validate local environment and model artifact loading.
3. Run automated tests.
4. Deploy a separate staging app from the feature branch.
5. Verify diagnosis, registration/login, both contribution flows, GridFS, dashboard values, and persistence after reboot.
6. Merge the tested release into `main`.
7. Configure rotated production database credentials in Streamlit Cloud Secrets.
8. Verify the existing production app and its feature flows.

### Production links

- Public repository: https://github.com/atharva315/Disease-Akinator
- Production app: https://disease-akinator-we7wpcyvg2tc4niefueebu.streamlit.app/
- Production merge PR: https://github.com/atharva315/Disease-Akinator/pull/2

## 8. Validation status

The project handoff records:

- 31 local automated tests passed.
- Staging verification of symptom prediction and skin-image classification.
- Staging verification of doctor registration/login, symptom contributions, image contributions/GridFS, and dashboard statistics.
- Staging data remained available after an application reboot.
- Production deployment and the same feature-level smoke checks were subsequently confirmed by the project owner.

These are deployment validation results, not proof of clinical validity or a substitute for the missing model-training evaluation report.

## 9. Known limitations and next documentation gate

The public repository now documents the deployed application from runtime input through storage and production deployment. **The original dataset-to-training process is only partially documented because its source datasets, preparation scripts, split manifests, training runs, and evaluation reports are not in the repository.**

The next documentation gate is to recover the original Kaggle/Colab training workspace or experiment exports and add the non-sensitive provenance/evaluation records listed in Section 5. Until then, the project is deployable and its inference path is documented, but training is not fully reproducible from a clean GitHub clone.
