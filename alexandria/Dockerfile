# Use the official Python slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# List Python version and location
RUN python --version && which python && which pip

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AZURE_INFERENCE_ENDPOINT="https://models.inference.ai.azure.com"
ENV AZURE_OPENAI_ENDPOINT="https://models.inference.ai.azure.com"
ENV AZURE_OPENAI_MODEL="gpt-4"
ENV AZURE_OPENAI_API_VERSION="2023-07-01-preview"
ENV PYTHONPATH=/app

# Expose the application port
EXPOSE 5000

# Set the command to run the application
CMD ["python", "app.py"]