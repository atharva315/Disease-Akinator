# Project Demonstration Video Guide

This guide is a recording plan for a concise 4–6 minute walkthrough of Disease Akinator. Use synthetic/non-identifying examples and do not show credentials, database connection strings, personal records, or private degree documents.

## Before recording

- Start the `disease-akinator-server` EC2 instance.
- Update DuckDNS and MongoDB Atlas Network Access if the EC2 public IP changed.
- Confirm `https://disease-akinator.duckdns.org` loads over HTTPS.
- Confirm the Docker container is healthy.
- Confirm Jenkins is accessible through the SSH tunnel.
- Open the GitHub repository, its CI workflow, the relevant merged release PR, and the Jenkins build page in separate tabs.
- Prepare a safe image example that you are permitted to use.
- Use a synthetic Doctor Portal profile and a synthetic degree PDF. Do not show real doctor credentials or patient data.
- Close terminals containing environment files, keys, passwords, or secret values.

## Suggested timeline

| Time | Segment | What to show |
|---|---|---|
| 0:00–0:25 | Introduction | Project name, problem it explores, and academic/educational scope |
| 0:25–1:10 | Symptom prediction | Start the questionnaire, answer a few questions, show the model output and alternatives |
| 1:10–1:50 | Skin-image classification | Upload a permitted demo image, classify it, and show the top-five scores |
| 1:50–2:30 | Doctor Portal | Show registration/login and the contribution dashboard using only synthetic test details; optionally demonstrate a test contribution |
| 2:30–3:10 | Architecture | Show README architecture diagram and explain Streamlit, model artifacts, PostgreSQL, and MongoDB GridFS |
| 3:10–4:35 | CI/CD | Show the protected PR flow, passing GitHub Actions checks, Jenkins SCM-triggered build, test/model/health stages, and ECR commit tag |
| 4:35–5:05 | Production hosting | Show the public HTTPS page and explain Nginx, Docker, EC2, SSM, and managed external databases |
| 5:05–5:30 | Limitations and close | State that the prototype is not clinically validated and that model training evidence is only partially archived |

Adjust the timeline to fit the final recording length. Do not spend time showing every test case or every AWS console setting.

## What to say about each capability

### Symptom prediction

Explain that the app maps four qualitative answers to numeric feature values and uses the saved scikit-learn model's probability output. Make clear that the result is an experimental prediction.

### Skin-image classification

Explain the RGB 160 × 160 preprocessing path, TensorFlow/Keras inference, class mapping, and top-five display. Do not imply that the output can diagnose a skin condition.

### Doctor Portal

Explain the separation between profile/contribution metadata in PostgreSQL and binary PDF/image content in GridFS. Mention Argon2id password hashing. Explicitly say that uploaded credentials are not independently verified and contributions do not automatically retrain the models.

### CI/CD

Show the real automated release path:
1. Feature branch into develop.
2. Release pull request from develop into main.
3. GitHub Actions unit-tests.
4. Jenkins detects the main-branch SCM change.
5. Jenkins checks out Git LFS assets, tests, builds Docker, verifies model compatibility, and smoke-tests Streamlit.
6. Jenkins pushes a commit-SHA-tagged image to ECR.
7. Jenkins retrieves its runtime configuration from SSM and deploys the new container.
8. Docker health status becomes healthy.

Do not show secret parameter values or AWS access credentials.

## Recommended final statement

“Disease Akinator demonstrates a complete application lifecycle: model inference, a database-backed contribution workflow, automated validation, containerization, and a Jenkins-to-AWS production release. It remains an educational prototype; clinical validation and full training-data reproducibility are outside the current implementation.”
