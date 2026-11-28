# Contributing

Thank you for helping improve Disease Akinator. This repository contains an educational disease-prediction prototype and a production deployment. Contributions should preserve the medical-safety boundaries, data-handling expectations, and release workflow described below.

## Before opening a change

1. Read the [README](README.md), [Architecture](docs/ARCHITECTURE.md), [Database guide](docs/DATABASE.md), and [Security policy](SECURITY.md).
2. Search existing issues and pull requests to avoid duplicate work.
3. Keep the change focused. Separate unrelated refactors, model changes, and infrastructure changes.
4. Never commit database credentials, .env files, Streamlit secrets, private datasets, patient information, certificates, SSH keys, or generated production records.

## Branch and pull-request workflow

The project uses the following flow:

    feature/<short-name>  ->  develop  ->  main

- Create a feature branch from develop, for example feature/improve-upload-validation.
- Make the change and add/update tests and documentation as appropriate.
- Open a pull request from the feature branch into develop.
- Wait for the required unit-tests checks to pass and resolve review feedback.
- Merge the feature pull request into develop.
- When a release is ready, open a separate pull request from develop into main.
- Confirm the unit-tests checks pass and that the branch is up to date before merging.
- A merge into main is a production release event: Jenkins polls the repository and, while its EC2 host is running, builds, publishes, and deploys the release image.

Do not push directly to main or bypass required branch protection. Do not merge a production release while the Jenkins EC2 host is intentionally stopped if an automatic deployment is expected.

## Local development

Use Python 3.13.x and Git LFS:

    git lfs install
    git clone https://github.com/atharva315/Disease-Akinator.git
    cd Disease-Akinator
    git lfs pull
    py -3.13 -m venv .venv

Activate the environment, then install runtime dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

For direct Streamlit development, configure isolated PostgreSQL and MongoDB credentials in `.streamlit/secrets.toml` or export the supported environment variables in your shell; the app does not automatically load a `.env` file. For Docker, use `.env.example` as the template for a local `.env` file passed with `--env-file`. See the README. Prediction pages can run without database configuration; Doctor Portal operations require both databases.

## Checks to run

Run the environment and model-artifact checks:

    python verify_environment.py
    python verify_model_assets.py

Run the same focused unit-test modules used by CI:

    python -m pip install -r requirements-ci.txt
    python -m compileall -q app.py database services tests
    python -m pytest -q tests/test_auth_service.py tests/test_doctor_service.py tests/test_contribution_service.py tests/test_mongodb_config.py

The repository also contains database integration tests. Only run them with dedicated test databases and test credentials. Do not point integration tests at production.

For a Docker build check:

    docker build -t disease-akinator:dev .

## Code and documentation expectations

- Preserve the feature order and numeric answer mapping expected by feature_names.pkl and the deployed symptom model.
- Keep the image preprocessing contract aligned with the saved image model: RGB, 160 × 160, float32, values scaled to [0, 1].
- Keep model artifacts and their associated feature/class metadata in sync.
- Update database/schema.sql and docs/DATABASE.md together when schema changes are made.
- Update tests whenever service, validation, or persistence behaviour changes.
- Update deployment and CI/CD docs when Docker, Jenkins, GitHub Actions, AWS resources, environment variables, or release steps change.
- Do not state that doctor accounts are verified or that contributions retrain models; those workflows are not implemented.
- Keep model performance claims tied to archived evaluation evidence. If evidence is unavailable, label a figure as project-reported and state the limitation.
- Use clear names, small functions, explicit validation, and safe error messages. Do not log secrets or full database connection strings.

## Commit message suggestions

Use concise, descriptive prefixes:

- feat: new user-facing capability
- fix: bug fix
- test: tests or fixtures
- docs: documentation
- refactor: internal code organisation
- build: Docker/dependency changes
- ci: automation changes

## Pull-request checklist

Before requesting a merge, confirm:

- [ ] The change is scoped and described.
- [ ] Tests/checks relevant to the change have been run.
- [ ] Documentation is updated.
- [ ] No secret, credential, patient record, or restricted dataset is included.
- [ ] Model inputs and artifacts remain compatible, where applicable.
- [ ] Database integration tests use isolated test resources.
- [ ] The pull request does not bypass branch protection.
- [ ] A production-impacting change has an explicit deployment and rollback plan.
