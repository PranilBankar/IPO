FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 IPOPILOT_DATA_DIR=/data

# Install CPU-only PyTorch first (avoids pulling the huge CUDA build from PyPI)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch line in requirements.txt is skipped via --no-deps override)
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt | grep -v "^pytest" | grep -v "^#" > requirements.prod.txt \
    && pip install --no-cache-dir -r requirements.prod.txt

COPY ipopilot ./ipopilot
EXPOSE 8000
CMD ["uvicorn", "ipopilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
