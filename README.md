# AI interior image classifier
### An AI model that classifies images depicting apartment interiors and assigns them specific characteristics/attributes and the styles in which they were built.

## 📌 Program description

The program is a **worker that processes photos of apartments**, operating in a queue architecture (RabbitMQ).

Its tasks include:
- receiving analysis tasks from the RabbitMQ queue,
- downloading photos assigned to a specific apartment,
- classifying photos using AI models (CLIP + LoRA),
- determining:
  - whether the photo shows the interior or exterior,
  - the type of room (e.g., living room, kitchen, bathroom),
  - the interior style,
- updating the analysis results in the system via API,
- determining the **dominant style of the apartment** based on all its photos.

The program operates as an **independent worker instance**, thanks to which:
- multiple instances can work in parallel,
- processing can be easily scaled,
- failures of a single worker do not result in data loss.

All communication is asynchronous, and RabbitMQ is responsible for:
- queuing tasks,
- load balancing between workers,
- reassigning tasks in case of an error.


# 📖 Instrukcja uruchomienia kontenera:

## ☢️ Wymagania

Na maszynie lokalnej:

- **Docker Desktop (testowany na wersji 4.55)**
- **Cloudflared**

## 1. Dostęp do RabbitMQ (Cloudflare)

Przed uruchomieniem kontenera **MUSISZ uruchomić tunel** z linku https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/.

Dla zainstalowanego pliku (nazwanego np. cloudflared.exe) uruchamiamy terminal z folderu, w którym znajduje się plik i wpisujemy komendę: 
```
.\cloudflared.exe access tcp --hostname rabbitmq.matiko.ovh --listener localhost:5672
```
### **Ważne** - nie wyłączamy konsoli po wpisaniu komendy!!!

## 2. Pobieramy repozytorium i rozpakowujemy je.

## 3. Wchodzimy do folderu image_analyzer.

## 4. Otwieramy terminal z poziomu tego folderu i wpisujemy komendę:

```
docker compose up --build
```

## 5. GOTOWE - program uruchomi się automatycznie po zbudowaniu kontenera.