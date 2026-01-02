#!/usr/bin/env python3
"""
Prosty przykład odbierania i wysyłania wiadomości do RabbitMQ z analizą zdjęć.
Używa API zamiast bezpośredniego połączenia z bazą danych.
Wszystkie wartości zapisywane do bazy są w lowercase.
"""

import os
import json
import logging
import signal
import sys
from dotenv import load_dotenv
import pika
import torch
import torch.nn as nn
import clip
from PIL import Image
from collections import Counter
import requests
from io import BytesIO
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.getenv('RABBITMQ_DEFAULT_USER', 'rabbit_user')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_DEFAULT_PASS', 'ChangeMeRabbit!')

INPUT_QUEUE = 'poi_results'
OUTPUT_QUEUE = 'image_classification_results'

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.matiko.ovh/")
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 30))

# Konfiguracja modelu
USE_LORA = True
LORA_WEIGHTS_PATH = os.getenv('LORA_WEIGHTS_PATH', 'lora_models/comprehensive_lora.pth')

# Stałe dla analizy zdjęć - zachowane oryginalne wartości dla analizy
ROOM_TYPES = [
     'salon',
     'aneks_kuchenny',
     'pokoj_dzieciecy',
     'salon',
     'salon_dziecieca',
     'salon',
     'jadalnia',
     'lazienka',
     'klatka_schodowa',
     'schody_wewnetrzne',
     'pokoj_biurowy',
     'pokoj_gier'
]

PHOTO_TYPES = [
    'NOWOCZESNY',
    'WSPOLCZESNY',
    'RUSTYKALNY',
    'TRADYCYJNY',
    'FARMHOUSE',
    'INDYJSKI',
    'NADMORSKI',
    'KOBIECY',
    'MINIMALISTYCZNY',
    'TROPIKALNY',
    'OTHERS'
]

ALLOWED_STYLES = {
    "modern",
    "classic",
    "industrial",
    "scandinavian",
    "minimalist",
    "vintage",
    "other"
}

# Rozszerzone mapowanie stylów
STYLE_MAPPING = {
    "NOWOCZESNY": "modern",
    "WSPOLCZESNY": "modern",
    "MINIMALISTYCZNY": "minimalist",
    "TRADYCYJNY": "classic",
    "RUSTYKALNY": "vintage",
    "FARMHOUSE": "vintage",
    "INDUSTRIAL": "industrial",
    "INDUSTRIALNY": "industrial",
    "SKANDYNAWSKI": "scandinavian",
    "SCANDINAVIAN": "scandinavian",
    "SCANDINAVIAN_STYLE": "scandinavian",
    "SKANDINAVIAN": "scandinavian",
    "VINTAGE": "vintage",
    "VINTAGE_STYLE": "vintage",
    "KLASYCZNY": "classic",
    "CLASSIC": "classic",
    "CLASSICAL": "classic",
    "INDYJSKI": "other",
    "NADMORSKI": "other",
    "KOBIECY": "other",
    "TROPIKALNY": "other",
    "TROPICAL": "other",
    "OTHERS": "other",
    "OTHER": "other",
    "INNY": "other"
}

# Mapowanie stylów z angielskiego na polskie dla room_style
STYLE_TO_POLISH = {
    "MODERN": "nowoczesny",
    "CLASSIC": "tradycyjny",
    "INDUSTRIAL": "nowoczesny",
    "SCANDINAVIAN": "nowoczesny",
    "MINIMALIST": "minimalistyczny",
    "VINTAGE": "rustykalny",
    "OTHER": "others"
}

# Mapowanie angielskich nazw pokojów na polskie (w lowercase)
ROOM_MAPPING_PL = {
    "living room": "salon",
    "bedroom": "sypialnia",
    "kitchen": "kuchnia",
    "bathroom": "lazienka",
    "dining room": "jadalnia",
    "office": "pokoj_biurowy",
    "hall": "klatka_schodowa",
    "interior stairs": "schody_wewnetrzne",
    "game room": "pokoj_gier",
    "kitchen alcove": "aneks_kuchenny",
    "kids room": "pokoj_dzieciecy",
    "kid bedroom": "sypialnia_dziecieca",
    "children room": "pokoj_dzieciecy",
    "children bedroom": "sypialnia_dziecieca",
    "child room": "pokoj_dzieciecy",
    "child bedroom": "sypialnia_dziecieca",
    "nursery": "pokoj_dzieciecy",
    "staircase": "schody_wewnetrzne",
    "stairs": "schody_wewnetrzne",
    "corridor": "klatka_schodowa",
    "hallway": "klatka_schodowa",
    "entry": "klatka_schodowa",
    "entrance": "klatka_schodowa"
}

# Domyślny priorytet typów pomieszczeń (w lowercase)
ROOM_TYPE_PRIORITY = [
    "salon", "sypialnia", "kuchnia", "lazienka",
    "jadalnia", "pokoj_dzieciecy", "aneks_kuchenny",
    "pokoj_biurowy", "sypialnia_dziecieca", "pokoj_gier",
    "klatka_schodowa", "schody_wewnetrzne"
]

warnings.filterwarnings('ignore')


class APIManager:
    """Menadżer do komunikacji z API"""

    def __init__(self):
        self.base_url = API_BASE_URL.rstrip('/')
        self.auth = (API_USERNAME, API_PASSWORD)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.timeout = API_TIMEOUT

    def get_photos_by_apartment(self, apartment_id: int) -> List[Dict]:
        """Pobiera wszystkie zdjęcia dla apartamentu z API"""
        try:
            # 1. Najpierw pobierz apartament
            apartment_url = f"{self.base_url}/apartments/{apartment_id}"
            apartment_response = self.session.get(apartment_url, timeout=API_TIMEOUT)
            apartment_response.raise_for_status()
            apartment_data = apartment_response.json()

            # 2. Z listy photo_ids pobierz każde zdjęcie osobno
            photo_ids = apartment_data.get('photo_ids', [])
            photos = []

            for photo_id in photo_ids:
                try:
                    photo_url = f"{self.base_url}/photos/{photo_id}"
                    photo_response = self.session.get(photo_url, timeout=API_TIMEOUT)
                    photo_response.raise_for_status()
                    photos.append(photo_response.json())
                except requests.exceptions.RequestException as e:
                    logger.error(f"Błąd pobierania zdjęcia {photo_id}: {e}")
                    continue

            return photos

        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd API przy pobieraniu apartamentu {apartment_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd: {e}")
            return []

    def update_photo(self, photo_id: int, update_data: Dict) -> bool:
        try:
            url = f"{self.base_url}/photos/{photo_id}"

            # 1. Pobierz aktualne dane
            response = self.session.get(url)

            if response.status_code != 200:
                logger.error(f"Cannot fetch photo {photo_id}: {response.status_code}")
                return False

            current_data = response.json()

            # 2. Przygotuj payload - konwertuj wszystkie wartości do lowercase
            payload = {
                "apartment_id": current_data.get("apartment_id", 0),
                "link": current_data.get("link", ""),
                "style": self._to_lowercase(update_data.get("style")),
                "room_type": self._to_lowercase(update_data.get("room_type")),
                "room_style": self._to_lowercase(update_data.get("room_style")),
                "photo_type": self._to_lowercase(update_data.get("photo_type")) or "non-interior"
            }


            # 3. Wykonaj PUT
            put_response = self.session.put(url, json=payload)

            if put_response.status_code == 200:
                logger.info(f"✓ Photo {photo_id} updated")
                return True
            else:
                logger.error(f"✗ PUT {put_response.status_code}: {put_response.text}")
                # Dodaj więcej szczegółów
                if put_response.status_code == 422:
                    try:
                        error_details = put_response.json()
                        logger.error(f"Validation errors: {json.dumps(error_details, indent=2)}")
                    except:
                        pass
                return False

        except Exception as e:
            logger.error(f"Error: {e}")
            return False

    def _to_lowercase(self, value):
        """Konwertuje wartość do lowercase, obsługuje None i puste ciągi"""
        if value is None:
            return ""
        if isinstance(value, str):
            value = value.strip()
            if value.lower() == "none" or value == "":
                return ""
            return value.lower()
        return str(value).lower()

    def update_apartment_style(self, apartment_id: int) -> bool:
        """Aktualizuje styl apartamentu na podstawie zdjęć (wartości w lowercase)"""
        try:
            # Najpierw pobierz wszystkie zdjęcia apartamentu
            photos = self.get_photos_by_apartment(apartment_id)
            logger.info(f"Znaleziono {len(photos)} zdjęć dla apartamentu {apartment_id}")

            if not photos:
                logger.info(f"Brak zdjęć dla apartamentu {apartment_id}")
                return False

            # Zbierz style zdjęć (z wyłączeniem 'other'/'others')
            styles_list = []
            for photo in photos:
                style = photo.get('style')
                logger.debug(f"Zdjęcie {photo.get('id')} ma styl: {style}")
                if style and style.lower() not in ['other', 'others', '']:
                    styles_list.append(style.lower())  # Konwertuj do lowercase

            logger.info(f"Lista stylów znalezionych: {styles_list}")

            if not styles_list:
                logger.info(f"Brak stylów (poza 'other') dla apartamentu {apartment_id}")
                return False

            # Znajdź dominujący styl (już w lowercase)
            style_counter = Counter(styles_list)
            logger.info(f"Licznik stylów: {dict(style_counter)}")
            dominant_style = style_counter.most_common(1)[0][0]

            logger.info(f"Dominujący styl dla apartamentu {apartment_id}: {dominant_style}")

            # Najpierw pobierz aktualne dane apartamentu
            url = f"{self.base_url}/apartments/{apartment_id}"
            logger.info(f"Pobieram dane apartamentu z: {url}")

            response = self.session.get(url, timeout=API_TIMEOUT)

            if response.status_code != 200:
                logger.error(
                    f"Nie można pobrać danych apartamentu {apartment_id}: {response.status_code} - {response.text}")
                return False

            apartment_data = response.json()

            # Przygotuj payload zgodnie z dokumentacją API
            payload = {
                "source_url": apartment_data.get("source_url", ""),
                "price": apartment_data.get("price", 0),
                "currency": apartment_data.get("currency", ""),
                "room_num": apartment_data.get("room_num", 0),
                "footage": apartment_data.get("footage", 0),
                "price_per_m2": apartment_data.get("price_per_m2", 0),
                "city": apartment_data.get("city", ""),
                "description": apartment_data.get("description", ""),
                "photo_attractiveness": apartment_data.get("photo_attractiveness", 0),
                "student_attractiveness": apartment_data.get("student_attractiveness", 0),
                "single_attractiveness": apartment_data.get("single_attractiveness", 0),
                "dog_owner_attractiveness": apartment_data.get("dog_owner_attractiveness", 0),
                "universal_attractiveness": apartment_data.get("universal_attractiveness", 0),
                "family_attractiveness": apartment_data.get("family_attractiveness", 0),
                "poi_desc": apartment_data.get("poi_desc", ""),
                "price_desc": apartment_data.get("price_desc", ""),
                "size_desc": apartment_data.get("size_desc", ""),
                "style": self._to_lowercase(dominant_style)  # Zaktualizowany styl
            }

            # ale nie są w wymaganym payloadzie
            for key in apartment_data:
                if key not in payload:
                    payload[key] = apartment_data[key]

            # Wykonaj PUT z pełnym payloadem
            logger.info(f"Wysyłam PUT do {url}")
            put_response = self.session.put(url, json=payload, timeout=API_TIMEOUT)

            logger.info(f"Odpowiedź PUT: {put_response.status_code}")

            if put_response.status_code in [200, 201, 204]:
                logger.info(f"✓ Zaktualizowano styl apartamentu {apartment_id} na: {dominant_style}")
                return True
            else:
                logger.error(f"✗ PUT {put_response.status_code}: {put_response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd API przy aktualizacji stylu apartamentu {apartment_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd przy aktualizacji stylu apartamentu: {e}")
            return False


def normalize_style(style: str | None) -> str:
    """
    Normalizuje styl do ALLOWED_STYLES (w lowercase).
    """
    if not style:
        return "other"

    s = style.strip().upper()

    # Najpierw sprawdź bezpośrednie dopasowanie
    if s.lower() in ALLOWED_STYLES:
        return s.lower()

    # Sprawdź w mapowaniu
    if s in STYLE_MAPPING:
        return STYLE_MAPPING[s].lower()

    # Sprawdź częściowe dopasowania
    for key, value in STYLE_MAPPING.items():
        if key.upper() in s or s in key.upper():
            return value.lower()

    # Jeśli nadal nie znaleziono, spróbuj inteligentnie dopasować
    s_lower = style.strip().lower()

    style_keywords = {
        "modern": ["nowoczesny", "współczesny", "modern", "contemporary"],
        "classic": ["tradycyjny", "klasyczny", "classic", "traditional"],
        "industrial": ["industrial", "industrialny", "loft"],
        "scandinavian": ["skandynawski", "scandinavian", "nordic"],
        "minimalist": ["minimalistyczny", "minimalist", "minimal"],
        "vintage": ["rustykalny", "vintage", "rustic", "farmhouse", "retro"],
        "other": ["indyjski", "nadmorski", "kobiecy", "tropikalny", "tropical", "other"]
    }

    for allowed_style, keywords in style_keywords.items():
        for keyword in keywords:
            if keyword in s_lower:
                return allowed_style

    return "other"


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, linear_module: nn.Linear, rank=4, alpha=8):
        super().__init__()
        self.linear = linear_module
        in_dim = linear_module.in_features
        out_dim = linear_module.out_features
        self.lora = LoRALayer(in_dim, out_dim, rank=rank, alpha=alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


def replace_linears_with_lora(module: nn.Module, rank=4, alpha=8):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            replace_linears_with_lora(child, rank=rank, alpha=alpha)


class URLImageLoader:
    @staticmethod
    def load_image_from_url(url, timeout=10):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert('RGB')
        except Exception as e:
            logger.error(f"Błąd ładowania {url}: {e}")
            return None


class InteriorImageDetector:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.categories = [
            "interior of a room", "living room", "bedroom", "kitchen", "bathroom",
            "dining room", "office interior", "apartment interior", "house interior",
            "interior design", "home decor",
            "building exterior", "outside of building", "street view", "garden",
            "balcony", "terrace", "hallway", "corridor", "entrance", "blueprint",
            "architectural plan", "advertisement", "children room", "kids bedroom",
            "nursery", "staircase", "stairs", "game room", "playroom", "study room",
            "home office", "closet", "wardrobe", "laundry room", "utility room",
            "basement", "attic", "garage"
        ]
        with torch.no_grad():
            text_tokens = clip.tokenize(self.categories).to(self.device)
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        self.interior_indices = list(range(0, 11))
        self.exterior_indices = list(range(11, 15))
        self.special_indices = list(range(15, 20))

    def detect_room_type(self, image):
        """Wykrywa typ pokoju na podstawie obrazu"""
        if image is None:
            return None, 0.0

        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

                room_mapping = {
                    1: "living room", 2: "bedroom", 3: "kitchen", 4: "bathroom",
                    5: "dining room", 6: "office", 14: "garden", 15: "balcony",
                    16: "terrace", 17: "hall", 18: "hall", 19: "hall",
                    22: "children room", 23: "kids bedroom", 24: "nursery",
                    25: "staircase", 26: "stairs", 27: "game room", 28: "playroom",
                    29: "office", 30: "office"
                }

                best_idx = similarities[0].argmax().item()
                confidence = similarities[0, best_idx].item()

                if best_idx in room_mapping and confidence > 0:
                    return room_mapping[best_idx], confidence
                else:
                    for idx, conf in enumerate(similarities[0]):
                        if conf > 0.1 and idx in room_mapping:
                            return room_mapping[idx], conf.item()

                    return None, confidence

        except Exception as e:
            logger.error(f"Błąd wykrywania typu pokoju: {e}")
            return "other", 0.0

    def is_interior_image(self, image):
        if image is None:
            return False, 0.0, "non-interior"
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                interior_confidence = similarities[0, self.interior_indices].sum().item()
                exterior_confidence = similarities[0, self.exterior_indices].sum().item()
                special_confidence = similarities[0, self.special_indices].sum().item()
                print(interior_confidence)
                if interior_confidence > 0.2:
                    return True, interior_confidence, "interior"
                elif special_confidence > 0.15:
                    return False, special_confidence, "non-interior"
                else:
                    return False, exterior_confidence, "non-interior"
        except Exception as e:
            return False, 0.0, f"error: {e}"


class CachedInteriorAnalyzer:
    def __init__(self, use_lora=True, lora_weights_path=LORA_WEIGHTS_PATH, rank=4, alpha=8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Urządzenie: {self.device}")

        # 1️⃣ Załaduj model CLIP
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)

        # 2️⃣ Jeśli LoRA, to wstrzykuj do Linearów
        if use_lora:
            logger.info("Zamiana Linearów na LoRA...")
            replace_linears_with_lora(self.model, rank=rank, alpha=alpha)
            if lora_weights_path and os.path.exists(lora_weights_path):
                try:
                    checkpoint = torch.load(lora_weights_path, map_location=self.device)
                    logger.info(f"Wczytano wag LoRA: {len(checkpoint)}")
                    model_dict = self.model.state_dict()
                    for k, v in checkpoint.items():
                        if k in model_dict:
                            model_dict[k].copy_(v)
                    self.model.load_state_dict(model_dict)
                    logger.info("✓ Wagi LoRA załadowane do modelu")
                except Exception as e:
                    logger.error(f"✗ Nie udało się wczytać LoRA: {e}")

        # 3️⃣ Inicjalizacja detektora wnętrz
        self.detector = InteriorImageDetector(device=self.device)

        # 4️⃣ Style
        self.styles = PHOTO_TYPES + [
            "INDUSTRIAL", "SCANDINAVIAN", "VINTAGE",
            "CLASSIC", "MODERN", "MINIMALIST"
        ]
        self.styles = list(dict.fromkeys(self.styles))
        self._precompute_text_features()

    def _precompute_text_features(self):
        """Przygotuj cechy tekstowe dla stylów wnętrz"""
        logger.info("Przygotowywanie cech tekstowych...")
        with torch.no_grad():
            texts = []
            for style in self.styles:
                if style == "NOWOCZESNY":
                    texts.append("modern contemporary interior design with clean lines and minimal furniture")
                elif style == "WSPOLCZESNY":
                    texts.append("contemporary interior design with current trends and modern elements")
                elif style == "MINIMALISTYCZNY":
                    texts.append("minimalist interior with few furniture pieces and clean spaces")
                elif style == "TRADYCYJNY":
                    texts.append("traditional classic interior with ornate furniture and rich details")
                elif style == "RUSTYKALNY":
                    texts.append("rustic vintage interior with wood elements and cozy atmosphere")
                elif style == "INDUSTRIAL":
                    texts.append("industrial interior with exposed bricks, concrete and metal elements")
                elif style == "SCANDINAVIAN":
                    texts.append("Scandinavian interior with light wood, white walls and functional design")
                elif style == "VINTAGE":
                    texts.append("vintage retro interior with old furniture and nostalgic elements")
                elif style == "CLASSIC":
                    texts.append("classical interior with symmetry, columns and traditional elements")
                elif style == "MODERN":
                    texts.append("modern interior design with innovative materials and sleek surfaces")
                elif style == "FARMHOUSE":
                    texts.append("farmhouse rustic interior with country style and natural materials")
                elif style == "INDYJSKI":
                    texts.append("Indian ethnic interior with colorful fabrics and intricate patterns")
                elif style == "NADMORSKI":
                    texts.append("coastal seaside interior with blue colors and nautical elements")
                elif style == "KOBIECY":
                    texts.append("feminine interior with soft colors, flowers and delicate furniture")
                elif style == "TROPIKALNY":
                    texts.append("tropical interior with palm leaves, exotic plants and vibrant colors")
                else:
                    texts.append(f"interior in {style} style")

            tokenized = clip.tokenize(texts).to(self.device)
            self.text_features = self.model.encode_text(tokenized)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def _get_default_room_type(self, image_analysis: Dict[str, Any]) -> str:
        """Zwraca domyślny typ pokoju na podstawie analizy obrazu."""
        room_type_en = image_analysis.get('detected_room_type_en')
        if room_type_en and room_type_en.lower() in ROOM_MAPPING_PL:
            return ROOM_MAPPING_PL[room_type_en.lower()]

        style = image_analysis.get('style')
        if style:
            style_lower = style.lower()
            if 'kitchen' in style_lower or 'kuchnia' in style_lower:
                return 'kuchnia'
            elif 'bathroom' in style_lower or 'lazienka' in style_lower:
                return 'lazienka'
            elif 'bedroom' in style_lower or 'sypialnia' in style_lower:
                return 'sypialnia'
            elif 'living' in style_lower or 'salon' in style_lower:
                return 'salon'
            elif 'office' in style_lower or 'biurowy' in style_lower:
                return 'pokoj_biurowy'
            elif 'children' in style_lower or 'dzieciecy' in style_lower:
                return 'pokoj_dzieciecy'

        confidence = image_analysis.get('confidence', 0)
        if confidence > 0.3:
            for room_type in ROOM_TYPE_PRIORITY:
                if room_type in ['salon', 'sypialnia', 'kuchnia', 'lazienka']:
                    return room_type

        return 'salon'

    def analyze_image(self, image):
        """Analizuje pojedynczy obraz"""
        if image is None:
            return {
                'is_interior': False,
                'style': None,
                'room_type': None,
                'photo_type': 'non-interior',
                'confidence': 0.0
            }

        try:
            # 1️⃣ Wykryj czy to wnętrze i typ pokoju
            is_interior, interior_conf, detected_type = self.detector.is_interior_image(image)
            photo_type = "interior" if is_interior else "non-interior"

            # 2️⃣ Wykryj typ pokoju
            room_type_en, room_confidence = self.detector.detect_room_type(image)
            room_type_pl = ROOM_MAPPING_PL.get(room_type_en.lower()) if room_type_en else None

            if not is_interior:
                return {
                    'is_interior': False,
                    'style': None,
                    'room_style': None,
                    'room_type': room_type_pl,
                    'photo_type': 'non-interior',
                    'confidence': interior_conf
                }

            # 3️⃣ Analiza stylu wnętrza
            with torch.no_grad():
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarities_style = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

                top_k = 3
                top_indices = similarities_style[0].topk(top_k).indices.cpu().numpy()
                top_values = similarities_style[0].topk(top_k).values.cpu().numpy()

                best_style_idx = None
                best_style_confidence = 0

                for idx, confidence in zip(top_indices, top_values):
                    raw_style = self.styles[idx]
                    normalized = normalize_style(raw_style)  # Już w lowercase

                    if normalized != "other" and confidence > 0.2:
                        best_style_idx = idx
                        best_style_confidence = confidence
                        break

                if best_style_idx is None and len(top_indices) > 0:
                    best_style_idx = top_indices[0]
                    best_style_confidence = top_values[0]

                if best_style_idx is not None:
                    raw_style = self.styles[best_style_idx]
                    style = normalize_style(raw_style)  # W lowercase
                    # Konwertuj styl na polskie room_style (w lowercase)
                    room_style_key = style.upper()
                    room_style = STYLE_TO_POLISH.get(room_style_key, "others")

                    if best_style_confidence < 0.15:
                        if best_style_confidence < 0.1:
                            style = "other"
                            room_style = "others"
                else:
                    style = "other"
                    room_style = "others"
                    best_style_confidence = 0.0

            # 4️⃣ Zapewnij, że wnętrza mają room_type
            if is_interior and photo_type == "interior" and room_type_pl is None:
                image_analysis = {
                    'detected_room_type_en': room_type_en,
                    'style': style,
                    'confidence': best_style_confidence
                }
                room_type_pl = self._get_default_room_type(image_analysis)
                logger.info(f"Ustawiono domyślny room_type dla wnętrza: {room_type_pl}")

            return {
                'is_interior': True,
                'style': style,  # W lowercase
                'room_style': room_style,  # W lowercase
                'room_type': room_type_pl,  # Już w lowercase z mapowania
                'photo_type': photo_type,  # "interior" lub "non-interior"
                'confidence': best_style_confidence,
                'detected_room_type_en': room_type_en
            }

        except Exception as e:
            logger.error(f"Błąd analizy obrazu: {e}")
            return {
                'is_interior': False,
                'style': None,
                'room_type': None,
                'photo_type': 'non-interior',
                'confidence': 0.0
            }


class RabbitMQProcessor:
    """Procesor wiadomości RabbitMQ z analizą zdjęć."""

    def __init__(self):
        self.connection = None
        self.channel = None
        self.api_manager = APIManager()
        self.analyzer = CachedInteriorAnalyzer(
            use_lora=USE_LORA,
            lora_weights_path=LORA_WEIGHTS_PATH
        )

    def connect(self):
        """Połącz się z RabbitMQ."""
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)

            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials,
                socket_timeout=10,
                blocked_connection_timeout=30,
                heartbeat=600
            )
            logger.info(f"Łączenie z RabbitMQ: {RABBITMQ_HOST}:{RABBITMQ_PORT}")

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            self.channel.queue_declare(queue=INPUT_QUEUE, durable=True)
            self.channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
            self.channel.basic_qos(prefetch_count=1)

            logger.info(f"✓ Połączono z RabbitMQ: {RABBITMQ_HOST}:{RABBITMQ_PORT}")
            logger.info(f"✓ Nasłuchiwanie kolejki wejściowej: {INPUT_QUEUE}")
            logger.info(f"✓ Wysyłanie do kolejki wyjściowej: {OUTPUT_QUEUE}")

        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"✗ Błąd połączenia z RabbitMQ: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Nieoczekiwany błąd połączenia z RabbitMQ: {e}")
            raise

    def process_message(self, message_data: dict) -> dict:
        """Przetwórz wiadomość z analizą WSZYSTKICH zdjęć apartamentu."""
        logger.info(f"Przetwarzam wiadomość: {message_data}")

        if 'apartment_id' not in message_data:
            return {
                'apartment_id': None,
                'processed': False,
                'error': 'Brak apartment_id w wiadomości',
                'timestamp': datetime.now().isoformat()
            }

        apartment_id = message_data['apartment_id']

        try:
            # Pobierz zdjęcia przez API
            photos = self.api_manager.get_photos_by_apartment(apartment_id)
            logger.info(f"Znaleziono {len(photos)} zdjęć dla apartamentu {apartment_id}")

            if not photos:
                return {
                    'apartment_id': apartment_id,
                    'processed': False,
                    'error': f'Nie znaleziono zdjęć dla apartamentu: {apartment_id}',
                    'timestamp': datetime.now().isoformat()
                }

            stats = {
                'total_photos': len(photos),
                'processed_success': 0,
                'processed_failed': 0,
                'interior_photos': 0,
                'exterior_photos': 0
            }

            # Przetwarzaj każde zdjęcie
            for photo in photos:
                photo_id = photo.get('id')
                photo_url = photo.get('link')

                if not photo_id or not photo_url:
                    logger.warning(f"Brak ID lub URL dla zdjęcia: {photo}")
                    stats['processed_failed'] += 1
                    continue

                # Załaduj obraz z URL
                img = URLImageLoader.load_image_from_url(photo_url)
                if img is None:
                    logger.warning(f"Nie można załadować zdjęcia {photo_id}")
                    stats['processed_failed'] += 1
                    continue

                # Analizuj obraz
                analysis_result = self.analyzer.analyze_image(img)

                # Loguj szczegóły analizy
                logger.info(
                    f"Zdjęcie {photo_id}: "
                    f"room_type={analysis_result.get('room_type', 'None')}, "
                    f"photo_type={analysis_result.get('photo_type', 'None')}, "
                    f"style={analysis_result.get('style', 'None')}"
                )

                # Przygotuj dane do aktualizacji przez API
                update_data = {
                    'room_type': analysis_result['room_type'],
                    'photo_type': analysis_result['photo_type'],
                    'room_style': analysis_result['room_style'],
                    'style': analysis_result['style']
                }

                # Aktualizuj przez API (APIManager automatycznie konwertuje do lowercase)
                success = self.api_manager.update_photo(photo_id, update_data)

                # Zbieraj statystyki
                if analysis_result['is_interior']:
                    stats['interior_photos'] += 1
                else:
                    stats['exterior_photos'] += 1

                if success:
                    stats['processed_success'] += 1
                    logger.info(f"✓ Zdjęcie {photo_id} zaktualizowane przez API")
                else:
                    stats['processed_failed'] += 1
                    logger.warning(f"✗ Błąd aktualizacji zdjęcia {photo_id} przez API")

            # Aktualizuj styl apartamentu przez API (tylko jeśli były wnętrza)
            if stats['interior_photos'] > 0:
                self.api_manager.update_apartment_style(apartment_id)
                logger.info(f"Zaktualizowano styl apartamentu {apartment_id} przez API")
            else:
                logger.info(f"Brak zdjęć wnętrz dla apartamentu {apartment_id}")

            # Przygotuj odpowiedź
            return {
                'apartment_id': apartment_id,
                'processed': True,
                'stats': stats,
                'message': f'Przetworzono {stats["processed_success"]}/{stats["total_photos"]} zdjęć',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Błąd przetwarzania apartamentu {apartment_id}: {e}")
            return {
                'apartment_id': apartment_id,
                'processed': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def on_message(self, ch, method, properties, body):
        try:
            message_data = json.loads(body.decode('utf-8'))
            logger.info(f"Otrzymano wiadomość z apartment_id: {message_data.get('apartment_id')}")

            result = self.process_message(message_data)

            # Wysyłamy TYLKO apartment_id i status
            output_message = {
                "apartment_id": result.get("apartment_id"),
                "processed": result.get("processed"),
                "timestamp": datetime.now().isoformat()
            }

            if "error" in result:
                output_message["error"] = result["error"]

            self.channel.basic_publish(
                exchange='',
                routing_key=OUTPUT_QUEUE,
                body=json.dumps(output_message),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type='application/json'
                )
            )

            logger.info(f"Wysłano do {OUTPUT_QUEUE}: {output_message}")
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except json.JSONDecodeError as e:
            logger.error(f"Błąd parsowania JSON: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Błąd przetwarzania wiadomości: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def start(self):
        """Rozpocznij nasłuchiwanie wiadomości."""
        try:
            self.channel.basic_consume(
                queue=INPUT_QUEUE,
                on_message_callback=self.on_message,
                auto_ack=False
            )

            logger.info(f"🔄 Oczekiwanie na wiadomości z kolejki '{INPUT_QUEUE}'...")
            logger.info("Naciśnij CTRL+C aby zakończyć")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("⏹️ Zatrzymywanie na żądanie użytkownika...")
            self.stop()

    def stop(self):
        """Zatrzymaj i zamknij połączenie."""
        try:
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
                logger.info("✓ Zatrzymano konsumpcję wiadomości")

            if self.connection and self.connection.is_open:
                self.connection.close()
                logger.info("✓ Zamknięto połączenie z RabbitMQ")
        except Exception as e:
            logger.error(f"Błąd podczas zamykania: {e}")


def signal_handler(signum, frame):
    """Obsługa sygnału zakończenia."""
    logger.info("📶 Otrzymano sygnał zakończenia")
    sys.exit(0)


def main():
    """Główna funkcja aplikacji."""
    import argparse

    parser = argparse.ArgumentParser(description='Analizator zdjęć wnętrz z RabbitMQ')
    parser.add_argument('--test', action='store_true', help='Tryb testowy (bez RabbitMQ)')
    parser.add_argument('--apartment-id', type=int, help='ID apartamentu do testu')

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.test:
        # Tryb testowy - przetwarzanie bez RabbitMQ
        test_message = {
            'apartment_id': args.apartment_id or 1,
            'timestamp': datetime.now().isoformat()
        }

        processor = RabbitMQProcessor()
        logger.info(f"🧪 Testowanie przetwarzania dla apartment_id: {test_message['apartment_id']}")
        result = processor.process_message(test_message)
        logger.info(f"📊 Wynik: {json.dumps(result, indent=2)}")

    else:
        # Normalny tryb pracy z RabbitMQ
        processor = RabbitMQProcessor()
        try:
            processor.connect()
            processor.start()
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"❌ Nie można połączyć się z RabbitMQ: {e}")
            logger.info("ℹ️  Sprawdź połączenie sieciowe i konfigurację RabbitMQ")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Błąd aplikacji: {e}", exc_info=True)
            sys.exit(1)
        finally:
            processor.stop()


if __name__ == '__main__':
    main()