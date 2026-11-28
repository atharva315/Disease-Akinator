# Disease Akinator

[![GitHub Actions CI](https://github.com/atharva315/Disease-Akinator/actions/workflows/ci.yml/badge.svg)](https://github.com/atharva315/Disease-Akinator/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)

**Production application:** [https://disease-akinator.duckdns.org](https://disease-akinator.duckdns.org)  
**Source code:** [github.com/atharva315/Disease-Akinator](https://github.com/atharva315/Disease-Akinator)

> **Availability note:** The production application runs on an on-demand AWS EC2 instance and is intentionally stopped when it is not needed. The URL is unavailable while that instance is stopped. Its public IPv4 address can change after a stop/start; operators must then update the DuckDNS record and MongoDB Atlas network allowlist.
>
> **Medical disclaimer:** Disease Akinator is an academic/educational software prototype, not a medical device. Predictions and confidence scores are experimental model outputs, not diagnoses, calibrated clinical certainty, or treatment advice. Do not use the application to make healthcare decisions or delay professional medical care.

## Overview

Disease Akinator is a multimodal Streamlit application with two machine-learning inference paths and a database-backed Doctor Portal:

1. **Symptom → disease prediction:** an interactive questionnaire uses a saved scikit-learn probability model.
2. **Skin image → condition classification:** an uploaded image is preprocessed and passed to a TensorFlow/Keras image classifier.
3. **Doctor Portal:** doctors can register, log in, and submit structured symptom contributions and labelled skin images for later review.

The production application serves pre-trained model artifacts. It does not train models at startup, and Doctor Portal contributions do not automatically retrain or change either deployed model.

## Live application

- **AWS EC2 production deployment:** [disease-akinator.duckdns.org](https://disease-akinator.duckdns.org)
- **Legacy Streamlit Community Cloud deployment:** [open legacy deployment](https://disease-akinator-we7wpcyvg2tc4niefueebu.streamlit.app/)

The EC2 deployment is the current production target described by the CI/CD documentation. The legacy Streamlit-hosted deployment is separate and may not track the latest production release.

## Application features

### 1. Symptom-based prediction

- Presents symptoms from the saved feature list in sequence.
- Supports four answer levels: **Yes**, **Probably**, **Probably Not**, and **No**.
- Converts answers to `1.0`, `0.8`, `0.2`, and `0.0`, respectively.
- Builds a feature vector in the exact order stored in `feature_names.pkl`.
- Uses the classifier's `predict_proba` output to update the prediction after answers.
- Uses a configured confidence threshold of `0.87` and a maximum of 25 questions per round before offering a checkpoint.
- Displays the leading prediction and up to three alternatives.

### 2. Skin-image classification

- Accepts JPG/JPEG and PNG images.
- Converts the image to RGB, resizes it to `160 × 160`, scales pixel values to `[0, 1]`, and adds a batch dimension.
- Loads the saved TensorFlow/Keras model for inference.
- Maps output indices through `class_names.pkl`.
- Displays the top five predicted classes and their model scores.

The application UI labels the image model **ResNet50 (DermNet NZ)**. The original dataset revision, licensing record, exact train/validation split, and complete training configuration are not preserved in this repository; see the model-development documentation before making reproducibility claims.

### 3. Doctor Portal

- Doctor registration collects a profile, email, password, and medical degree PDF.
- Login verifies passwords stored as **Argon2id hashes**.
- Symptom-to-disease contributions are stored as structured PostgreSQL records.
- Labelled image contributions are stored in MongoDB Atlas GridFS, with metadata and the file reference stored in PostgreSQL.
- The dashboard reads contribution statistics from PostgreSQL.

**Portal limitations:** Registration and PDF upload do not independently verify a medical qualification or approve an account. Email verification, password reset, and a dedicated administrator approval/review interface are not implemented. Contributions are stored for later review/research and are not fed into model training automatically.

## System architecture

The production application is hosted on one EC2 instance. Nginx handles public HTTP/HTTPS traffic and proxies requests to the Streamlit Docker container bound to loopback. The application connects to external managed databases; neither database is hosted on the EC2 instance.

~~~mermaid
flowchart TB
    B[Browser] --> DNS[DuckDNS hostname]
    DNS --> N[Nginx reverse proxy on AWS EC2]
    N -->|HTTPS at Nginx; proxy to loopback| S[Streamlit app in Docker]
    S --> SM[Symptom model + feature schema]
    S --> IM[Skin image model + class labels]
    S --> DP[Doctor Portal services]
    DP --> PG[(Neon PostgreSQL)]
    DP --> MG[(MongoDB Atlas GridFS)]
    PG --- P[Doctor profiles and contribution metadata]
    MG --- F[Degree PDFs and contribution image binaries]
~~~

The CI/CD flow is documented in [CI/CD Workflow](docs/CI_CD.md). In summary, GitHub Actions checks pull requests, while Jenkins polls the repository, validates main, builds and verifies a Docker image, publishes the commit-tagged image to Amazon ECR, retrieves runtime configuration from AWS Systems Manager Parameter Store, and deploys the image on EC2.

## Technology stack

| Layer | Technologies |
|---|---|
| User interface | Python 3.13, Streamlit |
| Symptom inference | scikit-learn 1.6.1, Joblib, NumPy |
| Image inference | TensorFlow CPU 2.20.0, Keras, Pillow |
| Structured data | Neon PostgreSQL, Psycopg 3 |
| Binary file storage | MongoDB Atlas, PyMongo, GridFS |
| Password hashing | Argon2id (argon2-cffi) |
| Model artifact versioning | Git LFS |
| Automated checks | GitHub Actions, pytest |
| Image build and production release | Docker, Jenkins, Amazon ECR |
| Runtime configuration | AWS Systems Manager Parameter Store (SecureString) |
| Hosting and networking | AWS EC2, Nginx, DuckDNS, Let's Encrypt |

## Repository structure

~~~text
Disease-Akinator/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   ├── pull_request_template.md
│   └── workflows/ci.yml
├── database/
│   ├── mongodb.py
│   ├── postgres.py
│   └── schema.sql
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CI_CD.md
│   ├── DATABASE.md
│   ├── DEPLOYMENT.md
│   ├── DEMO_GUIDE.md
│   ├── MODEL_DEVELOPMENT.md
│   ├── OPERATIONS.md
│   └── PROJECT_LIFECYCLE.md
├── services/
│   ├── auth_service.py
│   ├── contribution_service.py
│   └── doctor_service.py
├── tests/
│   ├── service/configuration unit tests
│   └── database integration tests
├── app.py
├── Dockerfile
├── Jenkinsfile
├── requirements.txt
├── requirements-ci.txt
├── verify_environment.py
└── verify_model_assets.py
~~~

The five model/metadata files are stored at the repository root and tracked with Git LFS:

| Artifact | Purpose |
|---|---|
| `final_ensemble.pkl` | Serialized symptom prediction classifier/ensemble |
| `feature_names.pkl` | Ordered symptom feature schema |
| `imputer.pkl` | Retained imputer artifact; not called by the current live symptom-inference path |
| `skin_model.h5` | TensorFlow/Keras skin-image classifier |
| `class_names.pkl` | Output-index-to-skin-class mapping |

The original training datasets and full training workspaces are not stored in this application repository. See [Model Development Record](docs/MODEL_DEVELOPMENT.md) for documented provenance gaps and historical metric context.

## Run locally

### Prerequisites

- Python **3.13.x**.
- Git and Git LFS.
- Docker, if you want to run the container locally.
- Neon PostgreSQL and MongoDB Atlas access for Doctor Portal operations. The prediction pages can run without database credentials.

### 1. Clone the repository and download model artifacts

    git lfs install
    git clone https://github.com/atharva315/Disease-Akinator.git
    cd Disease-Akinator
    git lfs pull

Check that the five model files are real LFS objects rather than small pointer files:

    git lfs ls-files

### 2. Create and activate a virtual environment

**Windows PowerShell**

    py -3.13 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

**macOS / Linux**

    python3.13 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

### 3. Configure local database credentials

Copy `.env.example` to `.env` and replace the placeholders with credentials for your own isolated development databases. The `.env` file is ignored by Git.

    cp .env.example .env

The application reads:

- `DISEASE_AKINATOR_POSTGRES_URL`
- `DISEASE_AKINATOR_MONGODB_URI`
- `DISEASE_AKINATOR_MONGODB_DATABASE` (optional; defaults to `disease_akinator`)

Alternatively, local Streamlit development can use `.streamlit/secrets.toml`:

    [postgres]
    url = "postgresql://<user>:<password>@<host>/<database>?sslmode=require"

    [mongodb]
    uri = "mongodb+srv://<user>:<password>@<cluster-host>/?retryWrites=true&w=majority"
    database = "disease_akinator"

Never commit real connection strings, passwords, API keys, certificates, or database exports. Do not reuse production database credentials for local development or testing.

### 4. Prepare databases (Doctor Portal only)

1. Create a PostgreSQL database and a restricted application role.
2. Review and apply `database/schema.sql` with the database owner/admin role. The script grants the application role DML access and assumes the role names documented in the file.
3. Create a MongoDB Atlas database user with the access required for the `disease_akinator` database and GridFS buckets.
4. Restrict network access to the app host where possible.

See [Database Setup and Data Model](docs/DATABASE.md) for schema details and the cross-database upload lifecycle.

### 5. Run preflight checks and start Streamlit

    python verify_environment.py
    python verify_model_assets.py
    python -m pip install -r requirements-ci.txt
    python -m pytest -q tests/test_auth_service.py tests/test_doctor_service.py tests/test_contribution_service.py tests/test_mongodb_config.py
    python -m streamlit run app.py

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

`verify_environment.py` checks the Python/runtime setup, imports, syntax, model assets, and whether database configuration is present; it does not print secret values or test database connectivity. `verify_model_assets.py` loads the model artifacts and runs synthetic inference checks.

## Run with Docker

Build after Git LFS objects have been downloaded:

    docker build -t disease-akinator:local .

Run the application, providing a local `.env` file containing the variables above:

    docker run --rm --name disease-akinator-local -p 8501:8501 --env-file .env disease-akinator:local

Then open `http://localhost:8501`.

The Docker image uses Python 3.13 slim, installs the runtime dependencies, runs as a non-root `app` user, exposes Streamlit on port 8501, and defines a health check at `/_stcore/health`. The production container is published only on the EC2 loopback interface; Nginx is the public entry point.

## Tests and continuous integration

The repository has unit and database-integration test modules. The GitHub Actions `unit-tests` job currently:

1. Checks out the repository.
2. Sets up Python 3.13.
3. Installs `requirements-ci.txt`.
4. Compiles the application, database, service, and test Python packages.
5. Runs the focused service/configuration unit-test modules:
   - `tests/test_auth_service.py`
   - `tests/test_doctor_service.py`
   - `tests/test_contribution_service.py`
   - `tests/test_mongodb_config.py`

The same focused checks run in the Jenkins pipeline before a release image is published. Database integration tests are present in `tests/`; run them only after configuring dedicated test databases, never against production.

See [CI/CD workflow](docs/CI_CD.md), [Contributing guide](CONTRIBUTING.md), [Deployment guide](docs/DEPLOYMENT.md), and [Operations runbook](docs/OPERATIONS.md).

## Model results and limitations

Historical project records report approximately:

| Model path | Reported result | Interpretation |
|---|---:|---|
| Symptom → disease | ~91% accuracy | Historical project-reported figure; its exact evaluation report and test protocol are not archived here |
| Skin image → condition | ~82% validation accuracy | Historical project-reported validation figure; split manifest and evaluation report are not archived here |

These figures are not independently reproduced by the current deployment pipeline and are not clinical-performance claims. The deployment smoke test checks model loading and synthetic inference compatibility, not predictive quality on new patient data.

## Responsible use, privacy, and security

- Outputs can be incorrect. Confidence scores are model probabilities, not calibrated clinical certainty.
- Do not upload identifiable patient information or real patient records to this academic prototype.
- Use synthetic or appropriately consented material for demonstrations and testing.
- Doctor-entered profile details and degree uploads are not independently verified by a credential authority.
- The application does not provide a clinical approval workflow, email verification, password reset, or automatic model retraining from contributions.
- Keep secrets in local ignored configuration or AWS SSM SecureString. Never put secret values in GitHub, logs, screenshots, sample data, or issues.

See [Security Policy](SECURITY.md) and [Database Documentation](docs/DATABASE.md).

## Documentation index

| Document | Contents |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Runtime components, request/data flow, and deployment topology |
| [CI/CD](docs/CI_CD.md) | Branch strategy, GitHub Actions, Jenkins stages, ECR publishing, and release gates |
| [Deployment](docs/DEPLOYMENT.md) | Local Docker and AWS production deployment configuration |
| [Operations](docs/OPERATIONS.md) | Start/stop, restart, health checks, DNS, and troubleshooting |
| [Database](docs/DATABASE.md) | PostgreSQL schema, MongoDB GridFS buckets, environment variables, and upload lifecycle |
| [Model development](docs/MODEL_DEVELOPMENT.md) | Available training-code evidence, datasets, models, artifacts, and reported metrics |
| [Project lifecycle](docs/PROJECT_LIFECYCLE.md) | What is reproducible, what is deployed, and known provenance gaps |
| [Demo guide](docs/DEMO_GUIDE.md) | Suggested walkthrough for a project demonstration video |

## License

No open-source license is currently specified. Until a license is added by the repository owner, the repository is source-available for viewing and does not grant permission to reuse, modify, or redistribute the code or model artifacts.
