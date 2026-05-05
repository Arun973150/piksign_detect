FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY piksign_detect/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY piksign_detect /app/piksign_detect
COPY detection/reverse-SynthID /app/detection/reverse-SynthID

WORKDIR /app/piksign_detect

ENV PIKSIGN_HOST=0.0.0.0 \
    PIKSIGN_PORT=8000 \
    PYTHONPATH=/app/piksign_detect:/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3); sys.exit(0)" || exit 1

CMD ["uvicorn", "frontend.app:app", "--host", "0.0.0.0", "--port", "8000"]
