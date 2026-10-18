FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 IPOPILOT_DATA_DIR=/data
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ipopilot ./ipopilot
EXPOSE 8000
CMD ["uvicorn", "ipopilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
