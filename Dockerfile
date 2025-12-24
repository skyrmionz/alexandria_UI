FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the alexandria folder
COPY alexandria/requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# CRITICAL FIX: Install CPU-only Torch first.
# Standard Torch includes CUDA (GPU) support which adds ~5GB to the image size.
# Railway runs on CPU, so we don't need the GPU bloat.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
# Using --default-timeout to prevent connection drops during large downloads
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the alexandria application folder into /app
COPY alexandria/ .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the application port
EXPOSE 5000

CMD ["python", "app.py"]
