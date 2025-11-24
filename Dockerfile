FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY interior_dataset.json .

# Create directory for models (CLIP will download here)
RUN mkdir -p /root/.cache/clip

# Create directory for output files
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache

# Default command
CMD ["python", "main.py", "--help"]