# main_v6_local_db.py
import os
import json
import torch
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
from collections import Counter
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import time
from pymongo import MongoClient
from datetime import datetime
import sys


class LocalDatabaseClient:
    def __init__(self, connection_string="mongodb://root:example@mongo:27017/interior_analysis?authSource=admin"):
        self.client = MongoClient(connection_string)
        self.db = self.client.interior_analysis
        self.apartments = self.db.apartments
        self.images = self.db.images
        self.analysis_results = self.db.analysis_results

    def get_pending_apartments(self):
        """Pobiera mieszkania z nieprzetworzonymi zdjęciami"""
        pipeline = [
            {
                '$lookup': {
                    'from': 'images',
                    'let': {'apt_id': '$_id'},
                    'pipeline': [
                        {
                            '$match': {
                                '$expr': {'$eq': ['$apartment_id', '$$apt_id']},
                                'analysis_status': 'pending'
                            }
                        }
                    ],
                    'as': 'pending_images'
                }
            },
            {
                '$match': {
                    'pending_images.0': {'$exists': True}
                }
            },
            {
                '$project': {
                    '_id': 1,
                    'title': 1,
                    'pending_count': {'$size': '$pending_images'}
                }
            }
        ]

        return list(self.apartments.aggregate(pipeline))

    def get_apartment_with_images(self, apartment_id):
        """Pobiera mieszkanie ze wszystkimi zdjęciami"""
        apartment = self.apartments.find_one({'_id': apartment_id})
        if not apartment:
            return None

        images = list(self.images.find({
            'apartment_id': apartment_id,
            'analysis_status': 'pending'
        }))

        return {
            'id': apartment['_id'],
            'title': apartment.get('title', ''),
            'images': images
        }

    def update_image_analysis(self, image_id, room_type, style, confidence):
        """Aktualizuje wyniki analizy dla zdjęcia"""
        self.images.update_one(
            {'_id': image_id},
            {
                '$set': {
                    'room_type': room_type,
                    'style': style,
                    'analysis_status': 'completed',
                    'analysis_confidence': confidence,
                    'analyzed_at': datetime.now()
                }
            }
        )

    def save_apartment_analysis(self, apartment_id, analysis_result):
        """Zapisuje zbiorcze wyniki analizy mieszkania"""
        self.analysis_results.update_one(
            {'apartment_id': apartment_id},
            {
                '$set': {
                    'overall_style': analysis_result['overall_style'],
                    'room_distribution': analysis_result['room_distribution'],
                    'analyzed_images': analysis_result['interior_images'],
                    'total_images': analysis_result['total_images'],
                    'analysis_date': datetime.now(),
                    'confidence': analysis_result['overall_style']['confidence']
                }
            },
            upsert=True
        )

    def export_analysis_results(self, output_file="analysis_export.json"):
        """Eksportuje wyniki analizy do pliku JSON"""
        results = list(self.analysis_results.find())

        # Konwersja ObjectId na string
        for result in results:
            result['_id'] = str(result['_id'])
            if 'analysis_date' in result:
                result['analysis_date'] = result['analysis_date'].isoformat()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Wyeksportowano {len(results)} wyników do {output_file}")
        return output_file


# Pozostałe klasy (LoRALayer, InteriorImageDetector, StyleRoomAnalyzer)
# pozostają bez zmian z poprzedniej wersji, ale dostosowujemy główną klasę:

class DatabaseStyleRoomAnalyzer:
    def __init__(self, db_client, use_lora=False, lora_weights_path=None, device=None):
        self.db = db_client
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Inicjalizacja modeli (bez zmian)
        self.detector = InteriorImageDetector(device=self.device)
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)

        # LoRA setup (bez zmian)
        self.use_lora = False
        if use_lora:
            print("Aplikuję LoRA do modelu...")
            replaced = replace_linears_with_lora(self.model, rank=4, alpha=8)
            print(f"Zastąpiono warstwy Linear: {len(replaced)}")
            if lora_weights_path and os.path.exists(lora_weights_path):
                load_lora_weights_to_model(self.model, lora_weights_path, strict_match=False)
            self.use_lora = True

        # Definicje stylów
        self.styles = [
            "nowoczesny", "klasyczny", "skandynawski", "industrialny", "rustykalny",
            "glamour", "minimalistyczny", "retro", "boho", "farmhouse"
        ]
        self._precompute_style_features()

    def _precompute_style_features(self):
        """Precompute features dla stylów"""
        with torch.no_grad():
            style_texts = [f"wnętrze w stylu {style}" for style in self.styles]
            tokenized = clip.tokenize(style_texts).to(self.device)
            self.style_features = self.model.encode_text(tokenized)
            self.style_features = self.style_features / self.style_features.norm(dim=-1, keepdim=True)

    def analyze_apartment_from_db(self, apartment_id, batch_size=8, confidence_threshold=0.3):
        """Analizuje mieszkanie z bazy danych"""
        print(f"\n=== Analizowanie mieszkania {apartment_id} ===")

        # Pobierz dane z bazy
        apartment_data = self.db.get_apartment_with_images(apartment_id)
        if not apartment_data or not apartment_data.get('images'):
            print("Brak zdjęć do analizy")
            return None

        room_analyses = []
        valid_images = []

        # Analizuj każde zdjęcie
        for img_data in apartment_data['images']:
            img_url = img_data['url']
            try:
                img = self._load_image_from_url(img_url)
                if img is None:
                    continue

                # Detekcja wnętrza i typu pomieszczenia
                is_interior, conf, category, room_type = self.detector.is_interior_image(
                    img, confidence_threshold
                )

                if is_interior:
                    valid_images.append({
                        'db_id': img_data['_id'],
                        'image': img,
                        'room_type': room_type,
                        'detection_confidence': conf
                    })
                else:
                    print(f"  Odrzucono: {category}")
                    # Oznacz jako nie-wnętrze
                    self.db.update_image_analysis(
                        img_data['_id'], 'not_interior', 'unknown', 0.0
                    )

            except Exception as e:
                print(f"Błąd przetwarzania {img_url}: {e}")

        if not valid_images:
            print("Brak zdjęć wnętrz do analizy stylów")
            return None

        # Analiza stylów dla zdjęć wnętrz
        print(f"Analizowanie stylów dla {len(valid_images)} zdjęć wnętrz...")
        style_predictions = self._analyze_styles_batch(
            [img_data['image'] for img_data in valid_images],
            batch_size
        )

        # Zapisz wyniki do bazy i przygotuj agregację
        for i, img_data in enumerate(valid_images):
            if i < len(style_predictions):
                style_result = style_predictions[i]

                # Zapisz do bazy
                self.db.update_image_analysis(
                    img_data['db_id'],
                    img_data['room_type'],
                    style_result['style'],
                    style_result['confidence']
                )

                room_analyses.append({
                    'room_type': img_data['room_type'],
                    'style': style_result['style'],
                    'style_confidence': style_result['confidence'],
                    'detection_confidence': img_data['detection_confidence']
                })

        # Oblicz zbiorcze wyniki
        overall_style = self._calculate_dominant_style(room_analyses)
        room_distribution = self._calculate_room_distribution(room_analyses)

        result = {
            'apartment_id': apartment_id,
            'total_images': len(apartment_data['images']),
            'interior_images': len(room_analyses),
            'overall_style': overall_style,
            'room_distribution': room_distribution
        }

        # Zapisz zbiorczy wynik
        self.db.save_apartment_analysis(apartment_id, result)

        print(f"Zakończono analizę mieszkania {apartment_id}")
        print(f"Dominujący styl: {overall_style['style']} (confidence: {overall_style['confidence']:.2f})")

        return result

    def _load_image_from_url(self, url):
        """Ładuje obraz z URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Błąd ładowania URL {url}: {e}")
            return None

    def _analyze_styles_batch(self, images, batch_size=8):
        """Analizuje style dla batcha obrazów (bez zmian)"""
        # ... implementacja identyczna jak poprzednio ...
        pass

    def _calculate_dominant_style(self, room_analyses):
        """Oblicza dominujący styl (bez zmian)"""
        # ... implementacja identyczna jak poprzednio ...
        pass

    def _calculate_room_distribution(self, room_analyses):
        """Oblicza rozkład pomieszczeń (bez zmian)"""
        # ... implementacja identyczna jak poprzednio ...
        pass


# Główna funkcja przetwarzania
def process_apartments_pipeline(use_lora=False, lora_weights=None, max_apartments=None,
                                batch_size=8, confidence_threshold=0.3):
    """
    Główny pipeline przetwarzania danych z lokalnej bazy
    """
    print("=== URUCHAMIANIE LOCAL DATABASE PIPELINE ===")

    # 1. Połączenie z bazą
    print("Łączenie z lokalną bazą danych...")
    db_client = LocalDatabaseClient()

    # 2. Sprawdź mieszkania do przetworzenia
    pending_apartments = db_client.get_pending_apartments()
    if not pending_apartments:
        print("Brak mieszkań do przetworzenia")
        return

    print(f"Znaleziono {len(pending_apartments)} mieszkań do przetworzenia")

    if max_apartments:
        pending_apartments = pending_apartments[:max_apartments]
        print(f"Ograniczono do {max_apartments} mieszkań")

    # 3. Inicjalizacja analizatora
    analyzer = DatabaseStyleRoomAnalyzer(
        db_client=db_client,
        use_lora=use_lora,
        lora_weights_path=lora_weights
    )

    # 4. Przetwarzaj każde mieszkanie
    successful = 0
    for apt in pending_apartments:
        try:
            result = analyzer.analyze_apartment_from_db(
                apt['_id'],
                batch_size=batch_size,
                confidence_threshold=confidence_threshold
            )
            if result:
                successful += 1
                print(f"✓ Pomyślnie przetworzono mieszkanie {apt['_id']}")
            else:
                print(f"✗ Nie udało się przetworzyć mieszkania {apt['_id']}")
        except Exception as e:
            print(f"✗ Błąd przetwarzania mieszkania {apt['_id']}: {e}")

    # 5. Eksport wyników
    print(f"\n=== PODSUMOWANIE ===")
    print(f"Pomyślnie przetworzono: {successful}/{len(pending_apartments)} mieszkań")

    export_file = db_client.export_analysis_results()
    print(f"Wyniki wyeksportowane do: {export_file}")

    return export_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analiza mieszkań z lokalnej bazy danych')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--use-lora', action='store_true', help='Użyj LoRA')
    parser.add_argument('--lora-weights', type=str, help='Ścieżka do wag LoRA')
    parser.add_argument('--max-apartments', type=int, help='Maksymalna liczba mieszkań')
    parser.add_argument('--batch-size', type=int, default=8, help='Rozmiar batcha')
    parser.add_argument('--confidence', type=float, default=0.3, help='Próg pewności')

    args = parser.parse_args()

    if args.export_only:
        db = LocalDatabaseClient()
        db.export_analysis_results()
        sys.exit(0)

    # Uruchom pipeline
    process_apartments_pipeline(
        use_lora=args.use_lora,
        lora_weights=args.lora_weights,
        max_apartments=args.max_apartments,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence
    )