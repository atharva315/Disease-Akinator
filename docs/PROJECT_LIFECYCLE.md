# Disease Akinator — Dataset, Model, and Production Lifecycle

This document connects the model artifacts, inference paths, Doctor Portal data lifecycle, and current production deployment. It separates implementation details verified from the repository from model-training provenance that is not fully archived.

- Repository: https://github.com/atharva315/Disease-Akinator
- Current production application: https://disease-akinator.duckdns.org
- Legacy Streamlit-hosted deployment: https://disease-akinator-we7wpcyvg2tc4niefueebu.streamlit.app/
- Default branch: main

## 1. Product scope

Disease Akinator is a Streamlit educational prototype with three user-facing areas: symptom-based disease prediction, skin-image classification, and a Doctor Portal for registration, login, and labelled symptom/image contributions.

The production application is an inference and contribution application. It does not train either prediction model at startup and does not automatically retrain models from submitted contributions.

> **Medical limitation:** This is not a clinically validated medical device. Predictions and confidence scores are experimental model outputs, not diagnoses, calibrated clinical certainty, or treatment advice.

## 2. Dataset and training provenance

The public application repository contains the deployed model artifacts and the metadata needed by the inference code. It does not contain the complete original raw datasets, all training notebooks/scripts, dataset audit packages, split manifests, training configurations, or held-out evaluation reports.

The focused [Model Development Record](MODEL_DEVELOPMENT.md) documents the available training-code snapshot, historical project notes, known differences between those sources, and the metrics reported by the project.

### Current documented evidence

- The symptom model's available training-code snapshot records a CSV with a diseases target, a non-stratified 75/25 split with random state 42, and a soft-voting classifier defined using Gaussian Naive Bayes, Logistic Regression, and K-Nearest Neighbours. An older project summary describes a different five-model ensemble. The exact membership of the deployed serialized binary is not independently confirmed by the repository evidence.
- Historical image-model notes describe transfer learning and filtering/augmentation, while the current UI labels the served image model ResNet50 (DermNet NZ). The exact dataset revision/licence, class filtering, split sizes, training configuration, and checkpoint evidence are not retained.
- Historical project records report approximately 91% symptom accuracy and 82% image validation accuracy. The corresponding evaluation artifacts and exact test protocol are not available for independent reproduction.

These historical values must remain labelled project-reported. They are not proof of clinical performance.

### Reproducibility items still unavailable

For each model, the repository does not currently provide all of the following:
- Exact dataset URL, revision, license/terms, and attribution record.
- Dataset audit, exclusions, duplicates/corruption checks, and final label distribution.
- Complete train/validation/test split counts and leakage-control evidence.
- Versioned dataset preprocessing and augmentation scripts.
- Complete training code/configuration, random seeds, hyperparameters, checkpoint selection, and experiment logs.
- Evaluation report with held-out metrics, confusion matrix, per-class metrics, and reproducible test protocol.

Do not infer these details from the serialized model artifacts. If the original training workspace is recovered, add a non-sensitive dataset manifest, split summary, training configuration, and evaluation report. Do not upload restricted medical images or patient-identifying information to GitHub.

## 3. Model artifacts delivered to production

The following artifacts are stored at the repository root and tracked through Git LFS:

| Artifact | Runtime purpose |
|---|---|
| final_ensemble.pkl | Serialized symptom prediction classifier/ensemble |
| feature_names.pkl | Ordered symptom feature schema used to construct model inputs |
| imputer.pkl | Retained imputer artifact; the current live symptom inference path does not call it |
| skin_model.h5 | TensorFlow/Keras skin-image classifier |
| class_names.pkl | Output-index-to-skin-class mapping |

A clean clone requires Git LFS and git lfs pull before model loading. The small pointer files in Git are not the model weights themselves.

## 4. Current inference pipelines

These details are based on the production app.py and model verification script.

### 4.1 Symptom prediction

1. Load final_ensemble.pkl and feature_names.pkl with Joblib.
2. Create a zero-initialized one-row input vector with one position for every saved feature.
3. Select the next unasked feature in the saved feature-list order.
4. Map answers to numeric values:
   - Yes → 1.0
   - Probably → 0.8
   - Probably Not → 0.2
   - No → 0.0
5. Obtain probabilities using predict_proba.
6. A maximum class probability at or above the configured 0.87 threshold can complete the round; otherwise the application offers a checkpoint after 25 questions.
7. Display the leading class and up to three alternatives.

The saved feature order is part of the model contract. A different feature order or answer mapping would make inputs incompatible with the deployed model.

### 4.2 Skin-image classification

1. Accept a JPG/JPEG or PNG upload.
2. Decode using Pillow and convert to RGB.
3. Resize to 160 × 160 pixels.
4. Convert to a float32 array and divide by 255.0.
5. Add a batch dimension and call the Keras model in inference mode.
6. Map output indices using class_names.pkl.
7. Display the top five model scores.

The production runtime uses TensorFlow CPU 2.20.0 and loads the HDF5 model with compile=False. This is a runtime/serving choice; the exact library version used when the model was originally trained/exported is not preserved.

## 5. Doctor Portal and database lifecycle

The Doctor Portal stores submitted profile and contribution data. It does not independently verify qualifications, provide a clinical approval process, or update the prediction models.

### Neon PostgreSQL

The schema in database/schema.sql defines:
- doctors: profile fields, normalized unique email, Argon2id password hash, and degree-file reference/metadata.
- symptom_contributions: one parent record per symptom submission.
- symptom_contribution_items: ordered symptom entries linked to a parent submission.
- image_contributions: disease label, doctor reference, metadata, and MongoDB GridFS file ID.

The schema grants the restricted application role data-manipulation permissions; schema ownership is separate.

### MongoDB Atlas GridFS

Binary files are stored in two buckets:
- doctor_degrees: uploaded degree PDFs.
- contribution_images: labelled images submitted through the portal.

PostgreSQL retains structured data and GridFS file IDs. The application uses Argon2id for passwords, validates file types/signatures/sizes, and attempts compensating GridFS deletion when a subsequent PostgreSQL insert fails.

### Registration and upload limits

- Password minimum: 12 characters.
- Degree document: PDF, maximum 10 MiB.
- Contribution image: PNG/JPEG, maximum 5 MiB.
- The uploaded degree document is not independently validated by a regulator or credential authority.
- Email verification, password reset, and a separate administrator approval workflow are not implemented.
- Contributions do not automatically retrain the models.

See [Database Setup and Data Model](DATABASE.md) for detailed fields and setup steps.

## 6. Production architecture and release lifecycle

The current production deployment uses a dedicated AWS EC2 instance in ap-southeast-2.

~~~mermaid
flowchart TB
    Browser --> DuckDNS
    DuckDNS --> Nginx[Nginx HTTPS reverse proxy on EC2]
    Nginx --> App[Dockerized Streamlit app on 127.0.0.1:8501]
    App --> Models[Git LFS model artifacts]
    App --> PG[(Neon PostgreSQL)]
    App --> Mongo[(MongoDB Atlas GridFS)]
    GitHub[GitHub repository] --> Actions[GitHub Actions unit tests]
    GitHub --> Jenkins[Jenkins SCM polling on EC2]
    Jenkins --> ECR[Amazon ECR commit-SHA image]
    Jenkins --> SSM[AWS SSM SecureString]
    ECR --> Deploy[Production Docker container]
    SSM --> Deploy
~~~

### Production components

- Nginx receives public HTTP/HTTPS traffic and proxies requests to the Streamlit container.
- The application container binds to 127.0.0.1:8501; the public entry point is Nginx.
- DuckDNS maps disease-akinator.duckdns.org to the current EC2 public IPv4 address.
- Let's Encrypt/Certbot supplies HTTPS.
- Neon PostgreSQL and MongoDB Atlas are external managed data services.
- Amazon ECR stores images tagged with the full source commit SHA.
- AWS SSM Parameter Store stores the production container environment as SecureString.
- An EC2 instance role provides scoped ECR and SSM access.

The current design is a single-instance setup. It does not use ECS, Fargate, an Application Load Balancer, or a database server on the EC2 host.

### CI/CD release lifecycle

1. Developer pushes a feature branch and opens a PR into develop.
2. GitHub Actions compiles the Python packages and runs the selected unit-test modules.
3. A separate release PR moves develop into main.
4. Jenkins polls SCM at the configured five-minute cadence while its EC2 host is running.
5. Jenkins checks out Git LFS assets, compiles and tests, builds the Docker image, validates model loading/synthetic inference, and runs a Streamlit health smoke test.
6. For main, Jenkins pushes an ECR image tagged with the full Git commit SHA.
7. Jenkins reads the SSM SecureString parameter into a permission-restricted temporary environment file.
8. Jenkins pulls the release image, replaces the application container, and waits for the Docker health check.
9. If the new container does not become healthy, Jenkins attempts to restart the previous image when one is available.

See [CI/CD Workflow](CI_CD.md), [Deployment Guide](DEPLOYMENT.md), and [Operations Runbook](OPERATIONS.md).

## 7. Configuration and on-demand operations

The application reads these environment variables:
- DISEASE_AKINATOR_POSTGRES_URL
- DISEASE_AKINATOR_MONGODB_URI
- DISEASE_AKINATOR_MONGODB_DATABASE (optional; defaults to disease_akinator)

Local Streamlit configuration can alternatively use [postgres].url, [mongodb].uri, and optional [mongodb].database in .streamlit/secrets.toml.

Production values are stored in the AWS SSM SecureString parameter documented in the deployment guide. Secret values must never be added to GitHub or displayed in logs.

The EC2 instance is intentionally stopped when the application is not required. While stopped, the public app and Jenkins are unavailable, although GitHub Actions continues to operate. The instance's automatically assigned public IPv4 address can change after a restart. Operators must update DuckDNS and add the new IP as a /32 MongoDB Atlas Network Access entry before expecting the public app and Doctor Portal database access to work. EBS storage and other retained AWS resources may continue to incur charges.

## 8. Validation status

The project release has been manually validated at the application and deployment layers:
- Symptom prediction and skin-image classification were tested on the production app.
- Doctor Portal registration, login, dashboard statistics, and contribution submission were tested.
- GitHub Actions pull-request checks passed through the feature → develop → main flow.
- Jenkins automatically detected a main-branch SCM change, completed the pipeline, pushed the commit-tagged image to ECR, pulled it onto EC2, and deployed it.
- The deployed container inspection confirmed the intended Git-SHA-tagged image was running with Docker health status healthy.
- The public HTTPS application was checked after an EC2 stop/start and DNS/network access update.

These checks validate the software/deployment workflow. They do not prove clinical validity or independently reproduce the historical model accuracy values.

## 9. Known limitations and next provenance gate

The repository documents the serving path, application data lifecycle, and production deployment. The original dataset-to-training process remains only partially reproducible because the raw datasets, full preparation/training workspaces, split manifests, and evaluation reports are not included.

If the original training workspace is recovered, add the non-sensitive provenance and evaluation material described in MODEL_DEVELOPMENT.md. Until then, preserve the model metrics as project-reported results and do not claim independently reproducible or clinically validated performance.
