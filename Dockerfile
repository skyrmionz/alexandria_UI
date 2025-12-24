FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Tesseract OCR
# Added --no-install-recommends to reduce image size and install time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the alexandria folder
COPY alexandria/requirements.txt .

# Upgrade pip and install requirements
# INCREASED TIMEOUT: The build is timing out during pip install.
# We are increasing the default timeout to prevent this.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the alexandria application folder into /app
COPY alexandria/ .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the application port
EXPOSE 5000

CMD ["python", "app.py"]
