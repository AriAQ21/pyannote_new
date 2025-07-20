FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY diarization.py .
COPY batch_diarization.py .

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "diarization.py", "--help"]
