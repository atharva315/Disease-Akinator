# Disease Akinator runtime image
# Keep this aligned with the project's Python 3.13 / requirements.txt setup.

FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    HOME=/home/app

WORKDIR /app

# libgomp1 is required by common NumPy / scikit-learn / TensorFlow CPU wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /home/app app \
    && mkdir -p /home/app/.streamlit \
    && chown -R app:app /home/app

# Install the exact application dependency set before copying source files.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Model artifacts tracked with Git LFS must be downloaded before docker build.
COPY --chown=app:app . .

USER app

EXPOSE 8501

# Streamlit exposes this endpoint for container health checks.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
