# DogTranslator — CPU image for the pipeline + Gradio app (Python 3.11, fixes the local 3.14 issue).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch \
    YOLO_CONFIG_DIR=/cache/ultralytics \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# OpenCV (pulled in by ultralytics) needs these shared libs; fonts for meme text rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (much smaller than the default CUDA wheels), then the rest.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements.txt .
# Flaky networks: retry, long timeout, and a BuildKit download cache so a retried build resumes instead of restarting.
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install --retries 10 --timeout 120 -r requirements.txt

# Code + models/yolov8s.pt only — .dockerignore keeps data/ and artifacts out; they are mounted as volumes.
COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
