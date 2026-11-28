# Deployment Guide

This guide documents local Docker use and the current AWS EC2 production deployment. Production releases should go through the GitHub pull-request flow and Jenkins; avoid manual replacement of the live application container.

## 1. Environments

| Environment | Purpose | Notes |
|---|---|---|
| Local Python / Docker | Development and local testing | Uses developer-owned database credentials |
| GitHub Actions | Pull-request and branch checks | Does not deploy the application |
| Jenkins on EC2 | Build, verify, publish, and deploy `main` | Polls GitHub while the EC2 host is running |
| AWS EC2 production | Hosts Nginx and the Streamlit Docker container | On-demand instance; site unavailable while stopped |
| Neon PostgreSQL | Production structured database | External managed service |
| MongoDB Atlas | Production GridFS file store | External managed service |

## 2. Local Streamlit deployment

Follow the [README local setup](../README.md#run-locally):

1. Install Python 3.13.x, Git, and Git LFS.
2. Clone the repository and run `git lfs pull`.
3. Create a virtual environment and install `requirements.txt`.
4. Configure local database credentials using `.env` or `.streamlit/secrets.toml`.
5. Apply `database/schema.sql` to a development PostgreSQL database and configure a development MongoDB Atlas database if the Doctor Portal is required.
6. Run `verify_environment.py` and `verify_model_assets.py`.
7. Start Streamlit with `python -m streamlit run app.py`.

The prediction pages use the model artifacts. Doctor Portal database actions require both configured services.

## 3. Local Docker deployment

The Docker build expects the real model files to have been pulled from Git LFS before building.

    git lfs pull
    docker build -t disease-akinator:local .
    docker run --rm --name disease-akinator-local -p 8501:8501 --env-file .env disease-akinator:local

The container listens on port 8501. For local testing, publish that port to the host as shown. The production container uses a loopback-only host binding and is reached through Nginx instead.

## 4. AWS production topology

Current deployment components:

| Component | Purpose |
|---|---|
| AWS EC2 (`ap-southeast-2`) | Dedicated host for Jenkins, Docker, Nginx, and the application container |
| Docker | Runs the Streamlit app as `disease-akinator` |
| Nginx | Public reverse proxy; accepts HTTP/HTTPS and forwards to local Streamlit |
| DuckDNS | Maps `disease-akinator.duckdns.org` to the current EC2 public IPv4 address |
| Let's Encrypt / Certbot | TLS certificate and renewal |
| Amazon ECR repository `disease-akinator` | Stores commit-SHA-tagged production images |
| AWS SSM Parameter Store | Stores the production container environment as SecureString |
| EC2 instance profile | Grants the host scoped ECR and SSM access |
| Neon PostgreSQL | Stores Doctor Portal structured records |
| MongoDB Atlas | Stores uploaded files in GridFS |

The app container binds only to `127.0.0.1:8501`. Nginx is the public entry point and redirects HTTP to HTTPS. The current system does not use ECS, Fargate, a load balancer, or a database server on EC2.

## 5. Production configuration

The Jenkins deployment expects an SSM SecureString parameter at:

    /disease-akinator/production/container-env

The parameter contains the following variable names, with real values configured only in AWS:

- `DISEASE_AKINATOR_POSTGRES_URL`
- `DISEASE_AKINATOR_MONGODB_URI`
- `DISEASE_AKINATOR_MONGODB_DATABASE`

Jenkins retrieves it with decryption during the deployment stage, writes it to a temporary file with restrictive permissions, supplies it to Docker, and deletes the temporary file. The values are not part of this repository.

The EC2 instance profile must allow access to the application ECR repository and this SSM parameter. Do not add static AWS access keys to the repository or Jenkinsfile.

## 6. Production release procedure

1. Confirm the Jenkins EC2 host is running. If it is stopped, Jenkins cannot poll for a new release.
2. Create/complete feature work using the feature → develop workflow.
3. Open a release pull request from `develop` into `main`.
4. Wait for required GitHub Actions checks and ensure the branch is current.
5. Merge the release PR.
6. Jenkins detects the `main` change through SCM polling (configured in the Jenkinsfile at approximately five-minute intervals while Jenkins is available).
7. Jenkins checks out the source and Git LFS assets, runs syntax/unit tests, builds the Docker image, verifies both model assets with synthetic inference, and runs a Streamlit container health check.
8. For `main`, Jenkins tags the release with the full Git commit SHA and pushes it to Amazon ECR.
9. Jenkins retrieves the SSM environment, pulls the release image, replaces the running app container, and waits for its Docker health check.
10. Confirm the Jenkins build ends successfully and the deployed container is healthy.
11. Open the public application and smoke-test the relevant pages.

The Jenkins pipeline includes a rollback attempt to the previous image if the new container fails its health check. The initial deployment cannot roll back to an image that did not previously exist.

See [CI/CD Workflow](CI_CD.md) for exact stages and [Operations Runbook](OPERATIONS.md) for verification and recovery.

## 7. Domain and HTTPS

The public application hostname is:

    https://disease-akinator.duckdns.org

DuckDNS points to the EC2 public IPv4 address. Nginx handles the HTTPS endpoint and proxies requests to the local Streamlit port. Certbot manages the Let's Encrypt certificate.

The EC2 instance currently uses an automatically assigned public IPv4 address rather than a reserved Elastic IP. A stop/start can therefore assign a different public address. After every start:

1. Read the new public IPv4 address in the EC2 console.
2. Update the DuckDNS A record for `disease-akinator`.
3. Add the new address as a `/32` entry in MongoDB Atlas Network Access.
4. Wait for DNS changes to resolve, then verify the public HTTPS URL.

Do not place a current IP address, security group identifier, account ID, or database hostname containing credentials in this document.

## 8. Stopping production when idle

Stopping the dedicated EC2 instance is supported and is the intended cost-control approach when the application is not needed.

While stopped:
- The public application is offline.
- Jenkins is offline and cannot poll or deploy.
- GitHub Actions continues to run because it is hosted by GitHub.
- External Neon and MongoDB Atlas services remain separate from EC2.
- EBS storage and other retained AWS resources may continue to incur charges.

Use **Stop instance**, not **Terminate instance**. Termination can delete the host and its attached resources depending on their settings.

On the next start, allow EC2 status checks to pass, then perform the DuckDNS and MongoDB Atlas IP updates described above. Docker uses the `unless-stopped` restart policy for the app container, and Docker/Nginx/Jenkins are configured as system services enabled at boot; verify them using the operations runbook.

## 9. What this guide does not cover

- Rebuilding either ML model from raw source datasets.
- Clinical validation or approval of predictions.
- Database backups, disaster-recovery guarantees, or formal retention policy.
- A multi-instance high-availability architecture.
- Automated DNS/IP updates after EC2 restart.

Those require separate design and operational work before the project is used beyond an academic demonstration.
