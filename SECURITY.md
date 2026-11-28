# Security Policy

## Scope

Disease Akinator is an academic/educational prototype. It is not a clinically validated medical device and is not designed or approved for storing real patient health records or other protected health information.

The application includes:
- Streamlit user interface and model inference.
- Doctor registration and password authentication.
- PostgreSQL records for doctor profiles and contribution metadata.
- MongoDB Atlas GridFS storage for uploaded degree PDFs and contribution images.
- AWS-hosted production deployment and CI/CD automation.

## Reporting a vulnerability

Please do not publish exploitable vulnerability details, credentials, database URLs, personal information, uploaded documents, or proof-of-concept patient data in a public GitHub issue.

Use GitHub's private vulnerability reporting feature for this repository if it is enabled. If it is not available, contact the repository owner through a private channel before sharing technical details. Include the affected component, impact, reproduction steps that do not expose real data, and any suggested mitigation.

## Security practices for contributors and operators

- Keep local .env and .streamlit/secrets.toml files out of Git.
- Store production application configuration in AWS Systems Manager Parameter Store as SecureString; never echo its value in Jenkins logs.
- Use an EC2 instance role for AWS access instead of long-lived AWS access keys on the host.
- Restrict PostgreSQL and MongoDB users to the minimum database permissions needed by the application.
- Restrict MongoDB Atlas network access to the current deployment egress IP where feasible; do not use an unrestricted 0.0.0.0/0 rule for convenience.
- Keep SSH access restricted to trusted source IP addresses.
- Use HTTPS for the public application.
- Do not place secrets in issues, pull requests, sample files, screenshots, logs, or documentation.
- Use synthetic or properly permissioned material for tests and demonstrations. Do not upload identifiable patient information.
- Treat doctor profile fields and degree uploads as user-submitted claims; the application does not independently verify credentials.

## Authentication and uploads

- Passwords are hashed using Argon2id; plaintext passwords are not stored.
- Registration enforces a minimum password length of 12 characters.
- Degree uploads are restricted to PDF and a 10 MB maximum, with signature validation.
- Contribution images are restricted to JPEG/PNG and a 5 MB maximum, with signature validation.
- Cross-database upload flows attempt compensating deletion if the relational metadata write fails.

These controls reduce some risks but do not constitute a full security assessment, compliance certification, or guarantee that the application is suitable for sensitive medical data.

## Supported release

Security fixes should target the current main branch and be released through the normal protected pull-request process unless a private coordinated response requires otherwise.
