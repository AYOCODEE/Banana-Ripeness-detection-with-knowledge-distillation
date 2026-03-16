FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app.py .
COPY configs/ ./configs/

# Expose port
EXPOSE 8000

# Default: run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
