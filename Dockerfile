FROM python:3.13-slim

WORKDIR /app

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
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--log-level", "warning", "--reload", "--host", "0.0.0.0", "--port", "8000"]
