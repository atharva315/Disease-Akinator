# CI/CD Workflow

This document describes the GitHub Actions and Jenkins release pipeline currently configured in this repository.

## 1. Branch and release model

The intended branch flow is:

    feature/<name>  ->  develop  ->  main

| Branch / event | Purpose | Automation |
|---|---|---|
| Push to `feature/**` | Validate feature work | GitHub Actions `unit-tests` |
| Push to `develop` | Validate integrated development | GitHub Actions `unit-tests` |
| Pull request into `develop` | Validate a feature before integration | GitHub Actions `unit-tests` |
| Pull request into `main` | Validate a release candidate | GitHub Actions `unit-tests` |
| Merge/update on `main` | Production release event | GitHub Actions plus Jenkins main-branch pipeline |

Use a feature branch from `develop`, merge it into `develop` after checks pass, then open a separate release pull request from `develop` into `main`. The main branch is protected; do not bypass its required checks.

## 2. GitHub Actions workflow

Workflow file: `.github/workflows/ci.yml`  
Workflow name: `Disease Akinator CI`  
Job: `unit-tests`

### Triggers

- Push to branches matching `feature/**`.
- Push to `develop`.
- Pull request events (opened, synchronize, reopened, ready for review) targeting `develop` or `main`.
- Manual dispatch from the GitHub Actions interface.

### Job steps

1. Check out the repository.
2. Set up Python 3.13.
3. Install `requirements-ci.txt`.
4. Compile Python files in `app.py`, `database`, `services`, and `tests`.
5. Run the focused test modules:
   - `tests/test_auth_service.py`
   - `tests/test_doctor_service.py`
   - `tests/test_contribution_service.py`
   - `tests/test_mongodb_config.py`

The CI dependency file deliberately avoids installing the full TensorFlow runtime just to run service/configuration unit tests. The workflow has read-only repository contents permission.

### Test scope

Database integration tests are present in the repository, but the current GitHub Actions job does not run the entire integration suite. Run integration tests only with dedicated test PostgreSQL and MongoDB resources. Never point them at production.

## 3. Jenkins multibranch pipeline

Pipeline file: `Jenkinsfile`  
Jenkins multibranch project: `Disease Akinator GitHub`

The Jenkinsfile polls SCM using:

    H/5 * * * *

This is Jenkins polling, not a GitHub webhook. The Jenkins host must be running for polling and builds to take place. A stopped EC2 host cannot detect a new commit until it is started again.

### Jenkins stages

| Stage | What it does |
|---|---|
| Checkout source and Git LFS assets | Checks out the branch, installs local Git LFS filters, pulls LFS objects |
| Python syntax and unit tests | Uses a Python 3.13 Docker container, compiles the Python files, runs the same focused unit-test modules |
| Build Docker image | Builds the application image from `Dockerfile` |
| Verify model assets | Runs `verify_model_assets.py` inside the image; loads both models and checks synthetic inference compatibility |
| Run Streamlit container health check | Starts an isolated smoke-test container and waits for the Streamlit `/_stcore/health` endpoint to report healthy |
| Publish production image to Amazon ECR | Runs only for `main`; tags and pushes the image using the full Git commit SHA |
| Deploy production container on this EC2 | Runs only for `main`; retrieves runtime configuration from SSM, pulls the image, starts the application container, and verifies health |

The post section removes the temporary smoke-test container and local build tag. ECR retains the published release image.

## 4. Image naming and release traceability

The Jenkins pipeline creates a local build tag based on the Jenkins build number. For a `main` release it publishes an ECR tag based on the Git commit:

    <account-id>.dkr.ecr.ap-southeast-2.amazonaws.com/disease-akinator:<full-git-commit-sha>

This makes the production image traceable to the exact source commit. The account ID is derived at build time through the EC2 instance role; it is not hard-coded into the repository.

## 5. Production environment configuration

Jenkins retrieves the encrypted parameter:

    /disease-akinator/production/container-env

The value is read from AWS Systems Manager Parameter Store with decryption enabled, written to a temporary file with restrictive file permissions, passed to the application container, and removed after the deployment stage. The parameter contents must never be printed or committed.

Expected variable names:

- `DISEASE_AKINATOR_POSTGRES_URL`
- `DISEASE_AKINATOR_MONGODB_URI`
- `DISEASE_AKINATOR_MONGODB_DATABASE`

The PostgreSQL and MongoDB values point to external managed services. Do not store credentials in the Jenkinsfile, Dockerfile, GitHub Actions YAML, or repository files.

## 6. AWS access model

The EC2 host uses an instance profile/role rather than static AWS credentials in the pipeline. The role is scoped to the application ECR repository and the production SSM parameter path. Its required capabilities include:

- ECR authorization token retrieval.
- ECR repository description and image push/pull actions scoped to the Disease Akinator repository.
- SSM parameter retrieval with decryption for the production environment parameter.
- STS caller identity lookup to derive the ECR registry account ID.

Do not attach this role to the separate ResumeClassification EC2 instance.

## 7. Deployment and rollback behaviour

For a `main` release, Jenkins:

1. Retrieves the encrypted application environment file.
2. Verifies required PostgreSQL and MongoDB environment variable names are present without printing their values.
3. Pulls the exact commit-tagged image from ECR before replacing the currently running application container.
4. Records the previous container image reference.
5. Removes the previous container and starts the new image using the name `disease-akinator`.
6. Publishes the Streamlit container port only on `127.0.0.1:8501`.
7. Uses Docker's `unless-stopped` restart policy.
8. Waits for the container health check.
9. If the new container fails health checks, attempts to start the previous image and waits for rollback health.

The first deployment has no previous application image to roll back to. Subsequent deployments can attempt rollback when the prior image reference is available. A successful Jenkins build should end with `Finished: SUCCESS` and a healthy production container.

## 8. GitHub status warning

The configured Jenkins GitHub personal access token is read-only for repository discovery/checkout. Jenkins may display a warning that it could not update a GitHub commit status because the token lacks status-write access (HTTP 403). This warning does not, by itself, indicate a failed pipeline; inspect the stage results and final build status.

Do not broaden token permissions just to remove a cosmetic status warning unless commit-status publishing becomes an explicit requirement. Any permission change should follow least privilege.

## 9. Verified release example

The documentation change used to verify automatic deployment was merged into `main`. Jenkins Build #2 reported **Started by an SCM change**, completed successfully, pushed a Git-SHA-tagged image to the `disease-akinator` ECR repository, pulled it on EC2, and started the production container. The EC2 container inspection confirmed the commit-tagged image was running with Docker health status `healthy`.

This is a pipeline/deployment verification, not a new model training or model-accuracy evaluation.

## 10. Release checklist

Before merging a release PR into `main`:

- [ ] The feature PR has been merged into `develop`.
- [ ] The release PR targets `main` from `develop`.
- [ ] Required GitHub Actions checks pass.
- [ ] The release branch is up to date with `main`.
- [ ] The Jenkins EC2 host is running and able to poll GitHub.
- [ ] The production SSM parameter exists and contains the expected variable names.
- [ ] The EC2 instance role can access the scoped ECR repository and SSM parameter.
- [ ] Git LFS assets are present and `verify_model_assets.py` passes.
- [ ] The deployment reaches a healthy container state.
- [ ] The public application is checked after deployment.
