FROM python:3.13-slim

WORKDIR /app

# Install system dependencies required for opencv, torch, ffmpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    libopenblas-dev \
    ffmpeg \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# COPY ultralytics/default.yaml /usr/local/lib/python3.13/site-packages/ultralytics/cfg/
# Fix Ultralytics default.yaml file
# RUN curl -o /usr/local/lib/python3.13/site-packages/ultralytics/cfg/default.yaml https://raw.githubusercontent.com/ultralytics/ultralytics/refs/heads/main/ultralytics/cfg/default.yaml

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
