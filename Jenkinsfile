pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 90, unit: 'MINUTES')
    }

    triggers {
        // Poll only while Jenkins is running. A stopped on-demand EC2 cannot
        // receive webhooks or poll GitHub; start the host before a release.
        pollSCM('H/5 * * * *')
    }

    environment {
        AWS_REGION = 'ap-south-1'
        ECR_REPOSITORY = 'disease-akinator'
        PRODUCTION_ENV_PARAMETER = '/disease-akinator/production/container-env'
        IMAGE_TAG = "disease-akinator:${BUILD_NUMBER}"
        SMOKE_CONTAINER_NAME = "disease-akinator-smoke-${BUILD_NUMBER}"
        APP_CONTAINER_NAME = 'disease-akinator'
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
                    docker rm -f "$SMOKE_CONTAINER_NAME" >/dev/null 2>&1 || true

                    # No host port is published: this health check must not
                    # collide with the production container on 127.0.0.1:8501.
                    docker run --detach \
                        --name "$SMOKE_CONTAINER_NAME" \
                        "$IMAGE_TAG"

                    attempt=0
                    while [ "$attempt" -lt 60 ]; do
                        status="$(docker inspect --format='{{.State.Health.Status}}' "$SMOKE_CONTAINER_NAME" 2>/dev/null || true)"

                        if [ "$status" = "healthy" ]; then
                            echo "PASS: Streamlit container health check succeeded."
                            exit 0
                        fi

                        if [ "$status" = "unhealthy" ]; then
                            docker logs "$SMOKE_CONTAINER_NAME"
                            exit 1
                        fi

                        running="$(docker inspect --format='{{.State.Running}}' "$SMOKE_CONTAINER_NAME" 2>/dev/null || true)"
                        if [ "$running" != "true" ]; then
                            docker logs "$SMOKE_CONTAINER_NAME"
                            echo "FAIL: Streamlit smoke-test container exited."
                            exit 1
                        fi

                        attempt=$((attempt + 1))
                        sleep 3
                    done

                    docker logs "$SMOKE_CONTAINER_NAME"
                    echo "FAIL: Streamlit container did not become healthy in time."
                    exit 1
                '''
            }
        }

        stage('Publish production image to Amazon ECR') {
            when {
                branch 'main'
            }
            steps {
                script {
                    env.RELEASE_IMAGE_URI = sh(
                        returnStdout: true,
                        script: '''
                            set -eu
                            account_id="$(aws sts get-caller-identity --query Account --output text)"
                            printf '%s.dkr.ecr.%s.amazonaws.com/%s:%s' \
                                "$account_id" "$AWS_REGION" "$ECR_REPOSITORY" "$GIT_COMMIT"
                        '''
                    ).trim()
                }
                sh '''
                    set -eu
                    registry="${RELEASE_IMAGE_URI%%/*}"
                    aws ecr describe-repositories \
                        --repository-names "$ECR_REPOSITORY" \
                        --region "$AWS_REGION" >/dev/null
                    aws ecr get-login-password --region "$AWS_REGION" \
                        | docker login --username AWS --password-stdin "$registry"
                    docker tag "$IMAGE_TAG" "$RELEASE_IMAGE_URI"
                    docker push "$RELEASE_IMAGE_URI"
                '''
            }
        }

        stage('Deploy production container on this EC2') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    set -eu
                    test -n "$RELEASE_IMAGE_URI"

                    # Fetch the encrypted runtime environment into a temporary,
                    # permission-restricted file. Never print its contents.
                    temp_env="$(mktemp)"
                    chmod 600 "$temp_env"
                    cleanup() {
                        rm -f "$temp_env"
                    }
                    trap cleanup EXIT

                    aws ssm get-parameter \
                        --name "$PRODUCTION_ENV_PARAMETER" \
                        --with-decryption \
                        --region "$AWS_REGION" \
                        --query 'Parameter.Value' \
                        --output text > "$temp_env"

                    grep -q '^DISEASE_AKINATOR_POSTGRES_URL=' "$temp_env"
                    grep -q '^DISEASE_AKINATOR_MONGODB_URI=' "$temp_env"

                    # Pull and verify the release image before touching the
                    # currently running application container.
                    docker pull "$RELEASE_IMAGE_URI"
                    previous_image="$(docker inspect --format='{{.Config.Image}}' "$APP_CONTAINER_NAME" 2>/dev/null || true)"

                    rollback_previous() {
                        echo "Attempting rollback to the previous application image."
                        docker rm -f "$APP_CONTAINER_NAME" >/dev/null 2>&1 || true

                        if [ -z "$previous_image" ]; then
                            echo "No previous application image was found; automatic rollback is unavailable."
                            return 1
                        fi

                        if ! docker run --detach \
                            --name "$APP_CONTAINER_NAME" \
                            --restart unless-stopped \
                            --publish 127.0.0.1:8501:8501 \
                            --env-file "$temp_env" \
                            "$previous_image"; then
                            echo "Rollback container could not be started."
                            return 1
                        fi

                        rollback_attempt=0
                        while [ "$rollback_attempt" -lt 60 ]; do
                            rollback_status="$(docker inspect --format='{{.State.Health.Status}}' "$APP_CONTAINER_NAME" 2>/dev/null || true)"
                            if [ "$rollback_status" = "healthy" ]; then
                                echo "Rollback succeeded: $previous_image"
                                return 0
                            fi

                            rollback_running="$(docker inspect --format='{{.State.Running}}' "$APP_CONTAINER_NAME" 2>/dev/null || true)"
                            if [ "$rollback_running" != "true" ] || [ "$rollback_status" = "unhealthy" ]; then
                                break
                            fi

                            rollback_attempt=$((rollback_attempt + 1))
                            sleep 3
                        done

                        docker logs "$APP_CONTAINER_NAME" || true
                        echo "Rollback container did not become healthy."
                        return 1
                    }

                    docker rm -f "$APP_CONTAINER_NAME" >/dev/null 2>&1 || true
                    if ! docker run --detach \
                        --name "$APP_CONTAINER_NAME" \
                        --restart unless-stopped \
                        --publish 127.0.0.1:8501:8501 \
                        --env-file "$temp_env" \
                        "$RELEASE_IMAGE_URI"; then
                        echo "FAIL: New production container could not be started."
                        rollback_previous || true
                        exit 1
                    fi

                    attempt=0
                    healthy=false
                    while [ "$attempt" -lt 60 ]; do
                        status="$(docker inspect --format='{{.State.Health.Status}}' "$APP_CONTAINER_NAME" 2>/dev/null || true)"
                        if [ "$status" = "healthy" ]; then
                            healthy=true
                            break
                        fi

                        running="$(docker inspect --format='{{.State.Running}}' "$APP_CONTAINER_NAME" 2>/dev/null || true)"
                        if [ "$running" != "true" ] || [ "$status" = "unhealthy" ]; then
                            break
                        fi

                        attempt=$((attempt + 1))
                        sleep 3
                    done

                    if [ "$healthy" = "true" ]; then
                        echo "PASS: Production container is healthy: $RELEASE_IMAGE_URI"
                        exit 0
                    fi

                    echo "FAIL: New production container failed its health check."
                    docker logs "$APP_CONTAINER_NAME" || true
                    rollback_previous || true
                    exit 1
                '''
            }
        }
    }

    post {
        always {
            sh '''
                docker rm -f "$SMOKE_CONTAINER_NAME" >/dev/null 2>&1 || true
                docker image rm "$IMAGE_TAG" >/dev/null 2>&1 || true
            '''
        }
    }
}
