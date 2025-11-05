FROM python:3.9-slim

# Ustawienie katalogu roboczego
WORKDIR /app

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Kopiowanie requirements i instalacja Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie kodu źródłowego
COPY . .

# Tworzenie katalogu na dane i modele
RUN mkdir -p lora_models
RUN mkdir -p data

# Zmienna środowiskowa dla Pythona
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Polecenie domyślne
CMD ["python", "main_v2.py", "--help"]