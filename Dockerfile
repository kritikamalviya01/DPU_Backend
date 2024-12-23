# Use Python 3.11-slim image
FROM python:3.11-slim

# Install production dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    portaudio19-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

# Expose the Flask app on port 8080
EXPOSE 8080

# Set environment variables for production
ENV FLASK_ENV=production
ENV FLASK_APP=app.py

# Use Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
