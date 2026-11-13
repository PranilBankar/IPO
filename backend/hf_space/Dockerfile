FROM python:3.11-slim

# HF Spaces runs containers as user 1000
RUN useradd -m -u 1000 user

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    IPOPILOT_DATA_DIR=/app/data \
    HOME=/home/user

# Install build deps (needed for some packages on HF's Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (no CUDA on HF free tier)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining production deps (skip torch & pytest)
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt | grep -v "^pytest" | grep -v "^#" > requirements.prod.txt \
    && pip install --no-cache-dir -r requirements.prod.txt

# Copy the app package
COPY --chown=user:user ipopilot ./ipopilot

# Create data directory with correct ownership
RUN mkdir -p /app/data && chown -R user:user /app

USER 1000

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "ipopilot.api.app:app", "--host", "0.0.0.0", "--port", "7860"]

