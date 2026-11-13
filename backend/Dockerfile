FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 IPOPILOT_DATA_DIR=/data

# Install build deps needed for some Python packages on ARM64
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch — ARM64 pulls the right wheel automatically from PyPI
# CPU-only: exclude CUDA index so we get the slim build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || \
    pip install --no-cache-dir torch  # fallback: PyPI ARM wheel

# Install remaining dependencies (skip torch & pytest lines)
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt | grep -v "^pytest" | grep -v "^#" > requirements.prod.txt \
    && pip install --no-cache-dir -r requirements.prod.txt

COPY ipopilot ./ipopilot
EXPOSE 8000
CMD ["uvicorn", "ipopilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
