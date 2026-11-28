# System Architecture

This document describes the implementation currently present in the repository and the production topology used by Disease Akinator. It distinguishes the application runtime from the build/release system.

## 1. High-level view

Disease Akinator has three user-facing areas: symptom-based prediction, skin-image classification, and the Doctor Portal. Prediction runs locally inside the Streamlit application using committed Git LFS model artifacts. Doctor Portal operations use two external data stores: Neon PostgreSQL for structured records and MongoDB Atlas GridFS for binary uploads.

~~~mermaid
flowchart TB
    User[Browser / User] --> DNS[DuckDNS]
    DNS --> Nginx[Nginx + HTTPS on EC2]
    Nginx --> Streamlit[Streamlit container :8501]
    Streamlit --> Symptom[scikit-learn symptom model]
    Streamlit --> Image[TensorFlow / Keras image model]
    Streamlit --> Services[Doctor Portal services]
    Services --> Postgres[(Neon PostgreSQL)]
    Services --> Mongo[(MongoDB Atlas GridFS)]
~~~

## 2. Application components

| Component | Repository location | Responsibility |
|---|---|---|
| Streamlit UI and page routing | `app.py` | Home page, symptom questionnaire, skin-image screen, Doctor Portal UI |
| Symptom inference | `app.py` + `final_ensemble.pkl` + `feature_names.pkl` | Build ordered numeric symptom vector and call `predict_proba` |
| Image inference | `app.py` + `skin_model.h5` + `class_names.pkl` | Preprocess uploaded image, run Keras inference, map output indices |
| Authentication | `services/auth_service.py` | Argon2id password hashing and verification |
| Doctor registration | `services/doctor_service.py` | Validate profile, hash password, coordinate degree PDF storage and profile creation |
| Contributions | `services/contribution_service.py` | Validate symptom/image contribution submissions and coordinate storage |
| PostgreSQL adapter | `database/postgres.py` | Structured records and contribution statistics |
| MongoDB adapter | `database/mongodb.py` | MongoDB client, GridFS buckets, file validation/upload/download/delete |
| Database schema | `database/schema.sql` | PostgreSQL tables, indexes, constraints and application-role grants |

## 3. Machine-learning inference paths

### Symptom → disease

1. The app loads `final_ensemble.pkl` and `feature_names.pkl` through Joblib.
2. It creates a zero-initialized one-row vector whose column order matches `feature_names.pkl`.
3. It asks the next unasked feature in the saved feature-list order.
4. It maps the user's answer as follows: Yes = 1.0, Probably = 0.8, Probably Not = 0.2, No = 0.0.
5. The app obtains class probabilities from `predict_proba`.
6. A configured probability threshold of 0.87 can complete the question round; otherwise a checkpoint appears after 25 questions.
7. The leading class and up to three alternatives are displayed.

The order and meaning of the saved feature schema are part of the model contract. Reordering features or changing the answer mapping without retraining and validating the model can invalidate predictions.

### Skin image → condition

1. The user uploads a JPG/JPEG or PNG.
2. Pillow decodes the image.
3. The app converts it to RGB, resizes to 160 × 160, converts to float32, scales pixel values by 1/255, and adds a batch dimension.
4. The Keras model is called in inference mode.
5. Output indices are mapped using `class_names.pkl`.
6. The UI shows the top five model scores.

The live UI identifies this model as ResNet50 (DermNet NZ). The public repository does not contain the complete training workspace, the exact dataset revision/licence record, or the full evaluation package. See `MODEL_DEVELOPMENT.md`.

## 4. Doctor Portal data flow

### Registration

- The service validates the submitted profile fields and PDF file.
- The password is converted to an Argon2id hash.
- The PDF is uploaded to the `doctor_degrees` GridFS bucket.
- A structured doctor record and the GridFS file reference are stored in PostgreSQL.
- If the PostgreSQL write fails after the PDF upload, the service attempts to delete the uploaded GridFS object.

### Login

- The email is normalized.
- The stored password hash is retrieved for verification.
- Argon2id verifies the password.
- The profile returned to the UI excludes the password hash.

### Symptom contribution

- A doctor submits a disease name, one or more symptoms, and optional notes.
- PostgreSQL stores a parent contribution and individual symptom rows.
- Contributions are stored for later review/research; the current application does not trigger model training.

### Image contribution

- The image is validated and uploaded to the `contribution_images` GridFS bucket.
- PostgreSQL stores the disease label, file ID, filename, content type, size, doctor reference, and optional notes.
- If the metadata insert fails, the service attempts a compensating delete of the newly uploaded GridFS object.

## 5. Production hosting topology

The live AWS deployment is intentionally simple:

- One dedicated EC2 instance hosts Jenkins, Docker, and Nginx.
- The Streamlit application runs in a Docker container named `disease-akinator`.
- The application container publishes port 8501 only to `127.0.0.1` on the EC2 host.
- Nginx listens on the public HTTP/HTTPS ports, redirects HTTP to HTTPS, and proxies requests to the local Streamlit port.
- Let's Encrypt/Certbot manages the TLS certificate.
- DuckDNS maps the public hostname to the EC2 public IPv4 address.
- Neon PostgreSQL and MongoDB Atlas remain external managed services.
- Amazon ECR stores commit-tagged release images.
- AWS Systems Manager Parameter Store stores encrypted runtime environment configuration.

No database server is hosted on the EC2 instance. The current topology does not use ECS, Fargate, an Application Load Balancer, or a NAT Gateway.

## 6. Delivery architecture

~~~mermaid
flowchart TD
    Feature[feature branch] -->|Pull request| Develop[develop]
    Develop -->|Release pull request| Main[main]
    Feature --> Actions[GitHub Actions]
    Develop --> Actions
    Main --> Actions
    Main -->|SCM polling while Jenkins host is running| Jenkins[Jenkins multibranch pipeline]
    Jenkins --> LFS[Checkout + Git LFS]
    LFS --> Tests[Syntax + focused unit tests]
    Tests --> Build[Docker build]
    Build --> Model[Model asset verification]
    Model --> Health[Container health smoke test]
    Health --> ECR[Amazon ECR commit-SHA image]
    ECR --> SSM[AWS SSM SecureString environment]
    SSM --> EC2[Deploy container on EC2]
    EC2 --> Nginx[Nginx / HTTPS]
~~~

GitHub Actions checks code changes and pull requests. Jenkins runs the release pipeline for `main`; the Jenkinsfile gates ECR publishing and production deployment with a `main` branch condition. The detailed triggers, IAM requirements, and rollback behaviour are in `CI_CD.md`.

## 7. Runtime configuration

The application supports these environment variables:

| Variable | Required for | Notes |
|---|---|---|
| `DISEASE_AKINATOR_POSTGRES_URL` | Doctor Portal | PostgreSQL connection string |
| `DISEASE_AKINATOR_MONGODB_URI` | Doctor Portal | MongoDB Atlas connection URI |
| `DISEASE_AKINATOR_MONGODB_DATABASE` | Doctor Portal | Optional; defaults to `disease_akinator` |

For local Streamlit development, the database modules also support `[postgres].url` and `[mongodb].uri` in `.streamlit/secrets.toml`. Production runtime values are supplied to Docker from AWS SSM SecureString; the values must never be committed or printed.

## 8. Important boundaries and limitations

- The product is an educational prototype, not a clinical decision-support system.
- Model probabilities are not calibrated medical certainty.
- Model assets are inference artifacts, not a reproducible training pipeline.
- Doctor registration and uploaded degree files are not independently verified by a credential authority.
- Contributions do not automatically retrain the models.
- No email verification, password reset, or administrator approval workflow is implemented.
- The production EC2 host is on-demand. When it is stopped, the application and Jenkins are unavailable; GitHub Actions remains independent and can still run.
