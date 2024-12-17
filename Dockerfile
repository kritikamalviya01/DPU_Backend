FROM python:3.11-slim

# Install only essential dependencies
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

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN python -m venv /env && \
    /env/bin/pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Set environment variables
ENV PATH="/env/bin:$PATH"
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Expose the Flask port
EXPOSE 8080

# Start Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]





# FROM python:3.11-slim

# RUN apt-get update && apt-get install -y \
#     cmake \
#     build-essential \
#     portaudio19-dev \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libglib2.0-dev \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# COPY requirements.txt /app/

# RUN python -m venv /env

# RUN /env/bin/pip install --no-cache-dir -r requirements.txt


# COPY . /app

# ENV PATH="/env/bin:$PATH"
# ENV FLASK_APP=app.py
# ENV FLASK_ENV=development

# EXPOSE 8080

# CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]


