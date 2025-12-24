FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the alexandria folder
COPY alexandria/requirements.txt .

# Upgrade pip and install requirements
# Using --no-cache-dir to keep image size down
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the alexandria application folder into /app
COPY alexandria/ .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the application port
EXPOSE 5000

CMD ["python", "app.py"]
