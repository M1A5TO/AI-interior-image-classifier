#!/bin/bash

# Skrypt do łatwego uruchamiania analizy wnętrz w Dockerze

set -e

# Sprawdzenie czy Docker jest zainstalowany
if ! command -v docker &> /dev/null; then
    echo "Error: Docker nie jest zainstalowany"
    exit 1
fi

# Funkcja pomocnicza
show_help() {
    echo "Użycie: $0 [OPCJE]"
    echo ""
    echo "OPCJE:"
    echo "  --train                    Przeprowadź trening LoRA"
    echo "  --analyze <obraz>          Analizuj obraz"
    echo "  --use-lora                 Użyj wytrenowanego modelu LoRA"
    echo "  --lora-weights <ścieżka>   Ścieżka do wag LoRA (domyślnie: lora_models/comprehensive_lora.pth)"
    echo "  --build                    Zbuduj obraz Dockera od nowa"
    echo "  --jupyter                  Uruchom Jupyter Lab"
    echo "  --help                     Pokaż tę pomoc"
    echo ""
    echo "PRZYKŁADY:"
    echo "  $0 --build --train"
    echo "  $0 --analyze test_images/living_room.jpg --use-lora"
    echo "  $0 --jupyter"
}

# Zmienne
BUILD=false
TRAIN=false
ANALYZE=""
USE_LORA=false
LORA_WEIGHTS="lora_models/comprehensive_lora.pth"
JUPYTER=false

# Parsowanie argumentów
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --train)
            TRAIN=true
            shift
            ;;
        --analyze)
            ANALYZE="$2"
            shift 2
            ;;
        --use-lora)
            USE_LORA=true
            shift
            ;;
        --lora-weights)
            LORA_WEIGHTS="$2"
            shift 2
            ;;
        --jupyter)
            JUPYTER=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Nieznana opcja: $1"
            show_help
            exit 1
            ;;
    esac
done

# Budowanie obrazu jeśli wymagane
if [ "$BUILD" = true ]; then
    echo "Budowanie obrazu Dockera..."
    docker-compose build interior-analyzer
fi

# Uruchamianie Jupytera
if [ "$JUPYTER" = true ]; then
    echo "Uruchamianie Jupyter Lab..."
    docker-compose --profile dev up jupyter
    exit 0
fi

# Trening
if [ "$TRAIN" = true ]; then
    echo "Uruchamianie treningu LoRA..."
    docker-compose run --rm interior-analyzer python main_v2.py --train --lora-weights "$LORA_WEIGHTS"
    exit 0
fi

# Analiza
if [ -n "$ANALYZE" ]; then
    echo "Analizowanie obrazu: $ANALYZE"

    # Sprawdzenie czy obraz istnieje
    if [ ! -f "$ANALYZE" ]; then
        echo "Error: Obraz $ANALYZE nie istnieje"
        exit 1
    fi

    # Przygotowanie komendy
    CMD="python main_v2.py --analyze /app/$ANALYZE"
    if [ "$USE_LORA" = true ]; then
        CMD="$CMD --use-lora --lora-weights $LORA_WEIGHTS"
    fi

    # Uruchomienie analizy
    docker-compose run --rm -v "$(pwd)/$(dirname "$ANALYZE"):/app/$(dirname "$ANALYZE")" interior-analyzer $CMD
    exit 0
fi

# Jeśli nie podano żadnych argumentów, pokaż pomoc
show_help