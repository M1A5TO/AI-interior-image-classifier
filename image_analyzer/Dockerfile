FROM python:3.11-slim

# Zapobiega problemom z logowaniem
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Systemowe zależności (Pillow, CLIP)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Najpierw requirements (lepszy cache Dockera)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Kod aplikacji
COPY app ./app

# Domyślny command
CMD ["python", "app/main.py"]
