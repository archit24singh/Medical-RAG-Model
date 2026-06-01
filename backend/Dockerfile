FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory structure
# Create bucket directory structure (mirrors local bucket/ folder)
# To upgrade to S3: remove this and configure boto3/s3fs in ingestion.py instead
RUN mkdir -p /app/bucket/patients /app/bucket/providers /app/bucket/uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
