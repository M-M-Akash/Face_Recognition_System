FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


FROM python:3.11-slim

# runtime libs for opencv GUI + ffmpeg for RTSP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    libgtk-3-0 libgdk-pixbuf-xlib-2.0-0\
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .

# Pre-download models at build time so the container works offline. The
# anti-spoofing weights are the same fetch src/antispoof.py does lazily at
# startup, so doing it here just makes that a no-op at runtime.
RUN python -c "from insightface.app import FaceAnalysis; \
    a = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); \
    a.prepare(ctx_id=-1, det_size=(640,640))" \
 && python -c "from src.antispoof import ensure_models; \
    assert len(ensure_models()) == 2, 'anti-spoofing weights failed to download'"

# No CMD here — the compose file decides what to run
