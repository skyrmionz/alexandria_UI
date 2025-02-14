# Dockerfile
FROM python:3.9-slim

# Update package lists and install Docker CLI
RUN apt-get update && apt-get install -y docker.io

WORKDIR /app

# Install any system dependencies if needed.
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code.
COPY . .

RUN ollama pull deepseek-r1:latest

EXPOSE 5000

CMD ["python", "app.py"]