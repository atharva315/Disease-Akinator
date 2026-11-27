pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        IMAGE_TAG = "disease-akinator:${BUILD_NUMBER}"
        CONTAINER_NAME = "disease-akinator-smoke-${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout source and Git LFS assets') {
            steps {
                checkout scm
                sh '''
                    set -eu
                    git lfs version
                    git lfs install --local
                    git lfs pull
                '''
            }
        }

        stage('Python syntax and unit tests') {
            steps {
                sh '''
                    set -eu
                    docker run --rm \
                        -v "$WORKSPACE:/workspace:ro" \
                        python:3.13-slim-bookworm \
                        sh -c '
                            set -eu
                            mkdir -p /tmp/disease-akinator-tests
                            cp -a /workspace/app.py /workspace/database /workspace/services /workspace/tests /workspace/requirements-ci.txt /tmp/disease-akinator-tests/
                            cd /tmp/disease-akinator-tests
                            python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-ci.txt
                            python -m compileall -q app.py database services tests
                            python -m pytest -q \
                                tests/test_auth_service.py \
                                tests/test_doctor_service.py \
                                tests/test_contribution_service.py \
                                tests/test_mongodb_config.py
                        '
                '''
            }
        }

        stage('Build Docker image') {
            steps {
                sh 'docker build --pull --tag "$IMAGE_TAG" .'
            }
        }

        stage('Verify model assets') {
            steps {
                sh 'docker run --rm "$IMAGE_TAG" python verify_model_assets.py'
            }
        }

        stage('Run Streamlit container health check') {
            steps {
                sh '''
                    set -eu
                    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
                    docker run --detach \
                        --name "$CONTAINER_NAME" \
                        --publish 127.0.0.1:8501:8501 \
                        "$IMAGE_TAG"

                    attempt=0
                    while [ "$attempt" -lt 60 ]; do
                        status="$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || true)"

                        if [ "$status" = "healthy" ]; then
                            echo "PASS: Streamlit container health check succeeded."
                            exit 0
                        fi

                        if [ "$status" = "unhealthy" ]; then
                            docker logs "$CONTAINER_NAME"
                            exit 1
                        fi

                        attempt=$((attempt + 1))
                        sleep 3
                    done

                    docker logs "$CONTAINER_NAME"
                    echo "FAIL: Streamlit container did not become healthy in time."
                    exit 1
                '''
            }
        }
    }

    post {
        always {
            sh '''
                docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
                docker image rm "$IMAGE_TAG" >/dev/null 2>&1 || true
            '''
        }
    }
}
