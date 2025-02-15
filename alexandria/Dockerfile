# Use the official Python slim image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (if needed for Docker-in-Docker scenarios)
# RUN apt-get install -y docker.io

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose the application port
EXPOSE 5000

# Set the command to run the application
CMD ["python", "app.py"]