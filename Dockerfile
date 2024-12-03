# Step 1: Use an official Python runtime as a parent image
FROM python:3.11-slim

# Step 2: Install system dependencies (CMake, build tools, libraries for OpenCV and TensorFlow)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    portaudio19-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set the working directory in the container
WORKDIR /app

# Step 4: Copy only the requirements file first (to take advantage of Docker layer caching)
COPY requirements.txt /app/

# Step 5: Create and activate a virtual environment
RUN python -m venv /env

# Step 6: Install the required Python packages from requirements.txt
RUN /env/bin/pip install --no-cache-dir -r requirements.txt

# Step 7: Copy the rest of the project files into the container
COPY . /app

# Step 8: Set environment variables
ENV PATH="/env/bin:$PATH"
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Step 9: Expose the port Flask will run on
EXPOSE 8080

# Step 10: Run the Flask app when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]





# # Step 1: Use an official Python runtime as a parent image
# FROM python:3.11-slim

# # Step 2: Install system dependencies (CMake, build tools, libraries for OpenCV and TensorFlow)
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
#     && rm -rf /var/lib/apt/lists/*

# # Step 3: Set the working directory in the container
# WORKDIR /app

# # Step 4: Copy the current directory contents into the container
# COPY . /app

# # Step 5: Install the required Python packages from requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Step 6: Expose the port Flask will run on
# EXPOSE 8080

# # Step 7: Define the environment variable for Flask
# ENV FLASK_APP=app.py

# # Step 8: Set the Flask environment to development for debugging
# ENV FLASK_ENV=development

# # Step 9: Run the Flask app when the container launches
# CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
