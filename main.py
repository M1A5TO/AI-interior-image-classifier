# main_v4_final.py
import os
import json
import torch
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
from collections import Counter
import pandas as pd
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')


# -------------------------
# LoRA Modules (bez zmian)
# -------------------------
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

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features


def replace_linears_with_lora(module: nn.Module, rank=4, alpha=8, replaced_names=None, parent_name=""):
    if replaced_names is None:
        replaced_names = []
    for name, child in list(module.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, nn.Linear):
            new_module = LoRALinear(child, rank=rank, alpha=alpha)
            setattr(module, name, new_module)
            replaced_names.append(full_name)
        else:
            replace_linears_with_lora(child, rank=rank, alpha=alpha, replaced_names=replaced_names,
                                      parent_name=full_name)
    return replaced_names


def save_lora_weights(model: nn.Module, path: str):
    sd = {}
    for name, param in model.named_parameters():
        if 'lora' in name:
            sd[name] = param.detach().cpu()
    torch.save(sd, path)
    print(f"Zapisano {len(sd)} parametrów LoRA do {path}")


def load_lora_weights_to_model(model: nn.Module, path: str, strict_match=False):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location='cpu')
    missing = []
    loaded = 0
    ckpt_keys = list(ckpt.keys())
    for name, param in model.named_parameters():
        if 'lora' not in name:
            continue
        if name in ckpt:
            param.data = ckpt[name].to(param.device)
            loaded += 1
            continue
        matched = None
        for k in ckpt_keys:
            if k.endswith(name) or name.endswith(k):
                matched = k
                break
        if matched:
            param.data = ckpt[matched].to(param.device)
            loaded += 1
        else:
            missing.append(name)
    print(f"Wczytano {loaded} LoRA parametrów z {path}. Brakujących: {len(missing)}")
    if strict_match and missing:
        raise RuntimeError(f"Nie wczytano wszystkich LoRA parametrów, brak: {missing[:10]}")
    return loaded, missing


# -------------------------
# Image loading helpers
# -------------------------
class URLImageLoader:
    @staticmethod
    def load_image_from_url(url, timeout=30):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert('RGB')
        except Exception as e:
            print(f"Błąd ładowania URL {url}: {e}")
            return None

    @staticmethod
    def load_images_from_csv(csv_path, max_images=None):
        try:
            df = pd.read_csv(csv_path)
            images = []
            for _, row in df.iterrows():
                images.append({'offer_id': row.get('offer_id', ''), 'seq': row.get('seq', ''), 'url': row['url']})
                if max_images and len(images) >= max_images:
                    break
            print(f"Załadowano {len(images)} URLi z CSV")
            return images
        except Exception as e:
            print(f"Błąd czytania CSV: {e}")
            return []


# -------------------------
# Interior Image Detector
# -------------------------
class InteriorImageDetector:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)

        # Kategorie do klasyfikacji - co zdjęcie może przedstawiać
        self.categories = [
            # Wnętrza - pozytywne
            "interior of a room", "living room", "bedroom", "kitchen", "bathroom",
            "dining room", "office interior", "apartment interior", "house interior",
            "interior design", "home decor",

            # Zewnętrza - negatywne
            "building exterior", "outside of building", "street view", "garden",
            "landscape", "cityscape", "outdoor",

            # Plany i diagramy
            "floor plan", "blueprint", "architectural plan", "diagram",
            "map", "technical drawing",

            # Logo i grafiki
            "company logo", "brand logo", "text", "signature",
            "advertisement", "brochure", "flyer",

            # Inne niepożądane
            "person", "people", "animal", "pet", "car", "vehicle",
            "close-up of object", "product photo", "furniture close-up"
        ]

        # Precompute text features dla wszystkich kategorii
        with torch.no_grad():
            text_tokens = clip.tokenize(self.categories).to(self.device)
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        # Definiujemy które kategorie są "wnętrzami"
        self.interior_indices = list(range(0, 11))  # pierwsze 11 to wnętrza
        self.non_interior_indices = list(range(11, len(self.categories)))

        print(
            f"Detektor wnętrz zainicjalizowany. Kategorie wnętrz: {len(self.interior_indices)}, inne: {len(self.non_interior_indices)}")

    def is_interior_image(self, image, confidence_threshold=0.3):
        """
        Sprawdza czy obraz przedstawia wnętrze mieszkania.
        Zwraca (is_interior: bool, confidence: float, top_category: str)
        """
        if image is None:
            return False, 0.0, "invalid image"

        try:
            # Preprocess i encode image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Oblicz podobieństwo do wszystkich kategorii
                similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

                # Znajdź najbardziej podobną kategorię
                top_conf, top_idx = similarities[0].topk(1)
                top_confidence = top_conf.item()
                top_category = self.categories[top_idx.item()]

                # Suma prawdopodobieństw dla kategorii "wnętrze"
                interior_confidence = similarities[0, self.interior_indices].sum().item()
                non_interior_confidence = similarities[0, self.non_interior_indices].sum().item()

                # Decyzja na podstawie confidence
                is_interior = interior_confidence > non_interior_confidence and top_confidence > confidence_threshold

                return is_interior, interior_confidence, top_category

        except Exception as e:
            print(f"Błąd podczas detekcji wnętrza: {e}")
            return False, 0.0, f"error: {str(e)}"


# -------------------------
# Analyzer z detekcją wnętrz
# -------------------------
class CachedInteriorAnalyzer:
    def __init__(self, use_lora=False, lora_weights_path=None, lora_rank=4, lora_alpha=8, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Urządzenie: {self.device}")

        # 1. Inicjalizacja detektora wnętrz
        self.detector = InteriorImageDetector(device=self.device)

        # 2. load base clip model
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)

        # 3. jeśli LoRA -> zastosuj przed precompute
        self.use_lora = False
        if use_lora:
            print("Aplikuję LoRA do modelu...")
            replaced = replace_linears_with_lora(self.model, rank=lora_rank, alpha=lora_alpha)
            print(f"Zastąpiono warstwy Linear: {len(replaced)}")
            if lora_weights_path and os.path.exists(lora_weights_path):
                print("Wczytywanie wag LoRA...")
                load_lora_weights_to_model(self.model, lora_weights_path, strict_match=False)
            else:
                print("Brak ścieżki do wag LoRA -> używam losowych LoRA")
            self.use_lora = True
        else:
            print("Nie używam LoRA - model bez modyfikacji")

        # 4. Load training data and precompute text features
        self.training_data = self._load_training_data()
        self.all_categories = self._extract_all_categories()
        self.text_features_cache = {}
        self._precompute_text_features_optimized()

    def _load_training_data(self, json_path="interior_dataset.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("training_data", [])
        except Exception as e:
            print(f"Nie udało się wczytać training data: {e}")
            return []

    def _extract_all_categories(self):
        styles = set()
        characteristics = set()
        materials = set()
        colors = set()
        room_types = set()
        for item in self.training_data:
            styles.add(item.get('style', ''))
            room_types.add(item.get('room_type', ''))
            for c in item.get('characteristics', []):
                characteristics.add(c)
            for m in item.get('materials', []):
                materials.add(m)
            for col in item.get('colors', []):
                colors.add(col)
        return {
            'styles': [s for s in styles if s],
            'characteristics': [c for c in characteristics if c],
            'materials': [m for m in materials if m],
            'colors': [c for c in colors if c],
            'room_types': [r for r in room_types if r]
        }

    def _precompute_text_features_optimized(self):
        print("Prekomputowanie cech tekstowych...")
        with torch.no_grad():
            for category, attributes in self.all_categories.items():
                if not attributes:
                    continue
                if category == 'room_types':
                    texts = [f"{a}" for a in attributes]
                else:
                    texts = [f"wnętrze z {a}" for a in attributes]

                tokenized = clip.tokenize(texts).to(self.device)
                text_features = self.model.encode_text(tokenized)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_features_cache[category] = text_features
        print("Prekomputowanie zakończone.")

    def filter_interior_images(self, image_paths, confidence_threshold=0.3):
        """
        Filtruje listę obrazów, zwracając tylko te które są wnętrzami
        """
        print(f" Filtrowanie {len(image_paths)} obrazów - wykrywanie wnętrz...")

        interior_images = []
        non_interior_info = []

        def process_image(p):
            try:
                if p.startswith('http'):
                    img = URLImageLoader.load_image_from_url(p)
                else:
                    img = Image.open(p).convert('RGB')

                if img is None:
                    return p, False, 0.0, "load error", None

                # Sprawdź czy to wnętrze
                is_interior, confidence, category = self.detector.is_interior_image(img, confidence_threshold)

                if is_interior:
                    return p, True, confidence, category, img
                else:
                    return p, False, confidence, category, None

            except Exception as e:
                print(f"Błąd przetwarzania {p}: {e}")
                return p, False, 0.0, f"error: {str(e)}", None

        # Przetwarzaj obrazy równolegle
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_image, image_paths))

        # Podziel wyniki
        for path, is_interior, confidence, category, img in results:
            if is_interior:
                interior_images.append((path, img, confidence))
            else:
                non_interior_info.append({
                    'path': path,
                    'confidence': confidence,
                    'category': category,
                    'reason': f"Nie wnętrze: {category} (confidence: {confidence:.3f})"
                })

        print(f" Znaleziono {len(interior_images)} obrazów wnętrz")
        print(f" Odrzucono {len(non_interior_info)} obrazów nie-wnętrz")

        # Wypisz informacje o odrzuconych
        for info in non_interior_info[:10]:  # Pierwszych 10
            print(f"    {info['path'][:50]}... -> {info['reason']}")
        if len(non_interior_info) > 10:
            print(f"   ... i {len(non_interior_info) - 10} więcej")

        return interior_images, non_interior_info

    def analyze_images_batch(self, image_paths, batch_size=16, filter_interiors=True, confidence_threshold=0.3):
        """
        Analizuje obrazy z opcjonalnym filtrowaniem wnętrz
        """
        results = {}
        valid_images = []
        image_metadata = []

        # Krok 1: Filtrowanie wnętrz jeśli wymagane
        if filter_interiors:
            interior_images, non_interior_info = self.filter_interior_images(image_paths, confidence_threshold)

            # Zapisz informacje o nie-wnętrzach do wyników
            for info in non_interior_info:
                results[info['path']] = {
                    'is_interior': False,
                    'interior_confidence': info['confidence'],
                    'detected_category': info['category'],
                    'analysis': {},
                    'reason': info['reason']
                }

            # Przygotuj obrazy do analizy
            for path, img, confidence in interior_images:
                valid_images.append(img)
                image_metadata.append({
                    'path': path,
                    'interior_confidence': confidence,
                    'is_interior': True
                })
        else:
            # Bez filtrowania - załaduj wszystkie obrazy
            print("  Pomijam filtrowanie wnętrz - przetwarzam wszystkie obrazy")
            for path in image_paths:
                try:
                    if path.startswith('http'):
                        img = URLImageLoader.load_image_from_url(path)
                    else:
                        img = Image.open(path).convert('RGB')

                    if img is not None:
                        valid_images.append(img)
                        image_metadata.append({
                            'path': path,
                            'interior_confidence': 1.0,  # Zakładamy że to wnętrze
                            'is_interior': True
                        })
                except Exception as e:
                    print(f"Błąd ładowania {path}: {e}")
                    results[path] = {
                        'is_interior': False,
                        'interior_confidence': 0.0,
                        'detected_category': 'load error',
                        'analysis': {},
                        'reason': f"Błąd ładowania: {str(e)}"
                    }

        if not valid_images:
            print("Brak obrazów do analizy")
            return results

        # Krok 2: Przetwarzanie obrazów w batchach
        print(f"  Przetwarzam {len(valid_images)} obrazów w batchach po {batch_size}...")

        # Przygotuj tensory
        tensors = []
        for img in valid_images:
            tensors.append(self.preprocess(img))

        all_features = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[i:i + batch_size]).to(self.device)
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_features.append(feats.cpu())

        image_features = torch.cat(all_features).to(self.device)

        # Krok 3: Analiza stylów dla każdego obrazu
        for idx, metadata in enumerate(image_metadata):
            feats = image_features[idx:idx + 1]
            analysis_results = {}

            for category, text_feats in self.text_features_cache.items():
                sims = (100.0 * feats @ text_feats.T).softmax(dim=-1)
                vals, inds = sims[0].topk(min(5, text_feats.shape[0]))
                attrs = self.all_categories[category]
                analysis_results[category] = [(attrs[i], v.item()) for v, i in zip(vals, inds)]

            results[metadata['path']] = {
                'is_interior': True,
                'interior_confidence': metadata['interior_confidence'],
                'detected_category': 'interior',
                'analysis': analysis_results,
                'reason': 'Success - interior image analyzed'
            }

        return results


# -------------------------
# Nowa funkcja do przetwarzania pliku zdjecia.txt
# -------------------------
def process_zdjecia_txt_file(input_file="zdjecia.txt", output_file=None,
                             use_lora=False, lora_weights=None,
                             filter_interiors=True, confidence_threshold=0.3,
                             batch_size=8):
    """
    Przetwarza plik zdjecia.txt, dodaje kolumny photo_type i room_type,
    oraz aktualizuje wartość style
    """
    print(f"Przetwarzanie pliku: {input_file}")

    # 1. Wczytaj dane z zdjecia.txt (JSON format)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Błąd parsowania pliku JSON: {e}")
        return None

    print(f"Wczytano {len(records)} rekordów")

    # 2. Utwórz listę URL-i do analizy
    urls = [record['link'] for record in records]

    # 3. Inicjalizuj analyzer
    analyzer = CachedInteriorAnalyzer(use_lora=use_lora, lora_weights_path=lora_weights,
                                      lora_rank=4, lora_alpha=8)

    # 4. Analizuj obrazy
    print("Rozpoczynam analizę obrazów...")
    results = analyzer.analyze_images_batch(
        urls,
        batch_size=batch_size,
        filter_interiors=filter_interiors,
        confidence_threshold=confidence_threshold
    )

    # 5. Zaktualizuj rekordy na podstawie wyników
    updated_records = []
    apartment_styles = {}  # Do zbierania stylów dla każdego mieszkania

    for idx, record in enumerate(records):
        url = record['link']
        result = results.get(url, {})

        # Pobierz room_type z analizy (jeśli dostępne)
        room_type = "unknown"
        if result.get('is_interior', False) and 'analysis' in result:
            room_types = result['analysis'].get('room_types', [])
            if room_types:
                # Weź pierwszy (najbardziej prawdopodobny) room_type
                room_type, room_confidence = room_types[0]
                if room_confidence < 0.3:  # Jeśli pewność niska
                    room_type = "unknown"

        # Pobierz detected style (jeśli dostępne)
        detected_style = record['style']  # domyślnie pozostawiamy oryginalny
        style_confidence = 0.0

        if result.get('is_interior', False) and 'analysis' in result:
            styles = result['analysis'].get('styles', [])
            if styles:
                # Weź pierwszy (najbardziej prawdopodobny) styl
                detected_style, style_confidence = styles[0]
                if style_confidence > 0.3:  # Jeśli pewność powyżej progu
                    record['style'] = detected_style

                    # Zapisz styl dla agregacji na poziomie mieszkania
                    apartment_id = record['apartment_id']
                    if apartment_id not in apartment_styles:
                        apartment_styles[apartment_id] = []
                    apartment_styles[apartment_id].append((detected_style, style_confidence))

        # Dodaj nowe kolumny
        record['photo_type'] = 'interior' if result.get('is_interior', False) else 'non-interior'
        record['room_type'] = room_type

        updated_records.append(record)

    # 6. Zapisz zaktualizowane rekordy
    if output_file is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"zdjecia_updated_{timestamp}.json"

    print(f"Zapisywanie zaktualizowanych danych do: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_records, f, ensure_ascii=False, indent=2)

    # 7. Stwórz podsumowanie
    interior_count = sum(1 for r in updated_records if r['photo_type'] == 'interior')
    style_changes = sum(1 for r in updated_records if r['style'] != 'other')

    print(f"\nPODSUMOWANIE:")
    print(f"  Przetworzono rekordów: {len(updated_records)}")
    print(f"  Zdjęcia wnętrz: {interior_count}")
    print(f"  Zdjęcia nie-wnętrz: {len(updated_records) - interior_count}")
    print(f"  Zmienionych stylów z 'other': {style_changes}")

    # Wyświetl przykładowe zmiany
    print(f"\nPrzykładowe zaktualizowane rekordy:")
    for i, record in enumerate(updated_records[:3]):  # Pierwsze 3 rekordy
        print(f"  Rekord {i + 1}:")
        print(f"    Oryginalny style: other")
        print(f"    Nowy style: {record['style']}")
        print(f"    Photo type: {record['photo_type']}")
        print(f"    Room type: {record['room_type']}")
        print()

    # 8. Zwróć dane potrzebne do przetworzenia mieszkania.json
    return {
        'input_file': input_file,
        'output_file': output_file,
        'total_records': len(updated_records),
        'interior_count': interior_count,
        'non_interior_count': len(updated_records) - interior_count,
        'style_changes': style_changes,
        'updated_records': updated_records,
        'apartment_styles': apartment_styles  # Ważne: styl dla każdego mieszkania
    }


# -------------------------
# Funkcja do przetwarzania pliku mieszkania.json
# -------------------------
def process_mieszkania_json_file(mieszkania_file="mieszkania.json",
                                 zdjecia_results=None,
                                 output_file=None):
    """
    Przetwarza plik mieszkania.json, aktualizując wartość 'style'
    na podstawie dominującego stylu ze zdjęć dla danego mieszkania.
    """
    print(f"Przetwarzanie pliku: {mieszkania_file}")

    # 1. Wczytaj dane z mieszkania.json
    try:
        with open(mieszkania_file, 'r', encoding='utf-8') as f:
            apartments = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Błąd parsowania pliku JSON: {e}")
        return None

    print(f"Wczytano {len(apartments)} mieszkań")

    # 2. Jeśli nie podano wyników z przetwarzania zdjęć, spróbuj wczytać domyślny plik
    if zdjecia_results is None:
        # Szukaj najnowszego pliku zdjecia_updated
        zdjecia_files = [f for f in os.listdir('.') if f.startswith('zdjecia_updated_') and f.endswith('.json')]
        if zdjecia_files:
            zdjecia_files.sort(reverse=True)
            latest_zdjecia = zdjecia_files[0]
            print(f"Wczytywanie najnowszego pliku ze zdjęciami: {latest_zdjecia}")
            with open(latest_zdjecia, 'r', encoding='utf-8') as f:
                zdjecia_data = json.load(f)

            # Oblicz styl dla każdego mieszkania na podstawie zdjęć
            apartment_styles = {}
            for zdjecie in zdjecia_data:
                apartment_id = zdjecie['apartment_id']
                if zdjecie['style'] != 'other' and zdjecie['photo_type'] == 'interior':
                    if apartment_id not in apartment_styles:
                        apartment_styles[apartment_id] = []
                    apartment_styles[apartment_id].append(zdjecie['style'])
        else:
            print("Nie znaleziono pliku z przetworzonymi zdjęciami")
            apartment_styles = {}
    else:
        # Użyj danych z przetwarzania zdjęć
        apartment_styles = zdjecia_results.get('apartment_styles', {})
        # Konwertuj format (styl, confidence) -> tylko styl
        simple_styles = {}
        for apt_id, styles_list in apartment_styles.items():
            simple_styles[apt_id] = [style for style, _ in styles_list]
        apartment_styles = simple_styles

    # 3. Oblicz dominujący styl dla każdego mieszkania
    apartment_dominant_styles = {}
    for apartment_id, styles_list in apartment_styles.items():
        if styles_list:
            # Znajdź najczęstszy styl
            style_counter = Counter(styles_list)
            most_common_style, count = style_counter.most_common(1)[0]
            total_styles = len(styles_list)
            confidence = count / total_styles if total_styles > 0 else 0

            apartment_dominant_styles[apartment_id] = {
                'style': most_common_style,
                'count': count,
                'total': total_styles,
                'confidence': confidence
            }

    print(f"\nObliczone dominujące style dla {len(apartment_dominant_styles)} mieszkań:")
    for apt_id, style_info in apartment_dominant_styles.items():
        print(f"  Mieszkanie {apt_id}: {style_info['style']} "
              f"({style_info['count']}/{style_info['total']} zdjęć, "
              f"pewność: {style_info['confidence']:.2f})")

    # 4. Zaktualizuj rekordy mieszkań
    updated_apartments = []
    style_updated_count = 0

    for apartment in apartments:
        apartment_id = apartment['id']

        # Sprawdź czy mamy styl dla tego mieszkania
        if apartment_id in apartment_dominant_styles:
            new_style = apartment_dominant_styles[apartment_id]['style']

            # Sprawdź czy styl się różni od obecnego (lub czy jest null)
            if apartment['style'] != new_style:
                apartment['style'] = new_style
                style_updated_count += 1
                print(f"  Zaktualizowano mieszkanie {apartment_id}: "
                      f"styl zmieniony na '{new_style}'")
            else:
                print(f"  Mieszkanie {apartment_id}: styl już ustawiony na '{new_style}'")
        else:
            print(f"  Mieszkanie {apartment_id}: brak danych o stylu ze zdjęć")

        updated_apartments.append(apartment)

    # 5. Zapisz zaktualizowane rekordy
    if output_file is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"mieszkania_updated_{timestamp}.json"

    print(f"\nZapisywanie zaktualizowanych danych do: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_apartments, f, ensure_ascii=False, indent=2)

    print(f"\nPODSUMOWANIE AKTUALIZACJI MIESZKAŃ:")
    print(f"  Przetworzono mieszkań: {len(updated_apartments)}")
    print(f"  Zaktualizowanych stylów: {style_updated_count}")
    print(f"  Mieszkań bez danych o stylu: {len(apartments) - len(apartment_dominant_styles)}")

    return {
        'input_file': mieszkania_file,
        'output_file': output_file,
        'total_apartments': len(updated_apartments),
        'style_updated_count': style_updated_count,
        'apartment_dominant_styles': apartment_dominant_styles,
        'updated_apartments': updated_apartments
    }


# -------------------------
# Funkcja do przetwarzania obu plików sekwencyjnie
# -------------------------
def process_both_files(zdjecia_input="zdjecia.txt",
                       mieszkania_input="mieszkania.json",
                       use_lora=False, lora_weights=None,
                       filter_interiors=True, confidence_threshold=0.3,
                       batch_size=8):
    """
    Przetwarza oba pliki sekwencyjnie:
    1. Przetwarza zdjęcia i aktualizuje style zdjęć
    2. Na podstawie zdjęć aktualizuje style mieszkań
    """
    print("=" * 60)
    print("PRZETWARZANIE ZDJĘĆ")
    print("=" * 60)

    # 1. Przetwórz zdjęcia
    zdjecia_results = process_zdjecia_txt_file(
        input_file=zdjecia_input,
        output_file=None,
        use_lora=use_lora,
        lora_weights=lora_weights,
        filter_interiors=filter_interiors,
        confidence_threshold=confidence_threshold,
        batch_size=batch_size
    )

    if zdjecia_results is None:
        print("Błąd przetwarzania zdjęć. Przerwano.")
        return None

    print("\n" + "=" * 60)
    print("PRZETWARZANIE MIESZKAŃ")
    print("=" * 60)

    # 2. Przetwórz mieszkania na podstawie wyników ze zdjęć
    mieszkania_results = process_mieszkania_json_file(
        mieszkania_file=mieszkania_input,
        zdjecia_results=zdjecia_results,
        output_file=None
    )

    print("\n" + "=" * 60)
    print("PODSUMOWANIE KOŃCOWE")
    print("=" * 60)
    print(f"1. Zdjęcia:")
    print(f"   - Przetworzono: {zdjecia_results['total_records']} zdjęć")
    print(f"   - Zdjęcia wnętrz: {zdjecia_results['interior_count']}")
    print(f"   - Zmienionych stylów: {zdjecia_results['style_changes']}")
    print(f"   - Wynik zapisano do: {zdjecia_results['output_file']}")

    if mieszkania_results:
        print(f"\n2. Mieszkania:")
        print(f"   - Przetworzono: {mieszkania_results['total_apartments']} mieszkań")
        print(f"   - Zaktualizowanych stylów: {mieszkania_results['style_updated_count']}")
        print(f"   - Wynik zapisano do: {mieszkania_results['output_file']}")

    return {
        'zdjecia_results': zdjecia_results,
        'mieszkania_results': mieszkania_results
    }


# -------------------------
# CLI and CSV handlers z detekcją wnętrz
# -------------------------
def analyze_images_from_csv(csv_path, use_lora=False, lora_weights=None, max_images=None,
                            batch_size=16, filter_interiors=True, confidence_threshold=0.3):
    images = URLImageLoader.load_images_from_csv(csv_path, max_images)
    urls = [d['url'] for d in images]

    analyzer = CachedInteriorAnalyzer(use_lora=use_lora, lora_weights_path=lora_weights,
                                      lora_rank=4, lora_alpha=8)

    results = analyzer.analyze_images_batch(urls, batch_size=batch_size,
                                            filter_interiors=filter_interiors,
                                            confidence_threshold=confidence_threshold)

    # Przygotuj wyniki z metadanymi ofert
    out = {}
    interior_count = 0
    non_interior_count = 0

    for d in images:
        url = d['url']
        key = f"{d['offer_id']}_{d['seq']}"

        if url in results:
            result_data = results[url]
            out[key] = {
                'url': url,
                'offer_id': d['offer_id'],
                'seq': d['seq'],
                'is_interior': result_data['is_interior'],
                'interior_confidence': result_data.get('interior_confidence', 0.0),
                'detected_category': result_data.get('detected_category', 'unknown'),
                'reason': result_data.get('reason', ''),
                'analysis': result_data.get('analysis', {})
            }

            if result_data['is_interior']:
                interior_count += 1
            else:
                non_interior_count += 1
        else:
            # Obraz nie został przetworzony
            out[key] = {
                'url': url,
                'offer_id': d['offer_id'],
                'seq': d['seq'],
                'is_interior': False,
                'interior_confidence': 0.0,
                'detected_category': 'not processed',
                'reason': 'Image not processed due to error',
                'analysis': {}
            }
            non_interior_count += 1

    # Agreguj wyniki na poziomie ofert
    offers_summary = aggregate_results_by_offer(out)

    # Przygotuj końcową strukturę wyników
    final_results = {
        'image_level_results': out,
        'offer_level_summary': offers_summary,
        'statistics': {
            'total_images': len(images),
            'interior_images': interior_count,
            'non_interior_images': non_interior_count,
            'unique_offers': len(offers_summary)
        }
    }

    # Zapisz wyniki
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    out_path = f"analysis_results_{timestamp}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n PODSUMOWANIE:")
    print(f"    Obrazy wnętrz: {interior_count}")
    print(f"    Obrazy nie-wnętrz: {non_interior_count}")
    print(f"    Unikalne oferty: {len(offers_summary)}")

    print(f"    Wyniki zapisano do {out_path}")

    return final_results


# -------------------------
# Helper function for offer-level aggregation
# -------------------------
def aggregate_results_by_offer(results):
    """Helper function kept for compatibility"""
    # Uproszczona wersja - w tym kontekście nieużywana
    return {}


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze-csv', type=str, help='csv with url column')
    parser.add_argument('--process-zdjecia', type=str, help='Process zdjecia.txt file', default="zdjecia.txt")
    parser.add_argument('--process-mieszkania', type=str, help='Process mieszkania.json file',
                        default="mieszkania.json")
    parser.add_argument('--process-both', action='store_true', help='Process both files sequentially')
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--lora-weights', type=str, default='lora_models/comprehensive_lora.pth')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--no-filter-interiors', action='store_true',
                        help='Przetwarzaj wszystkie obrazy, nawet nie-wnętrza')
    parser.add_argument('--confidence-threshold', type=float, default=0.3, help='Próg pewności dla detekcji wnętrz')
    parser.add_argument('--output-file-zdjecia', type=str, help='Nazwa pliku wyjściowego dla zdjęć')
    parser.add_argument('--output-file-mieszkania', type=str, help='Nazwa pliku wyjściowego dla mieszkań')

    args = parser.parse_args()

    if args.analyze_csv:
        analyze_images_from_csv(
            args.analyze_csv,
            use_lora=args.use_lora,
            lora_weights=args.lora_weights,
            max_images=args.max_images,
            batch_size=args.batch_size,
            filter_interiors=not args.no_filter_interiors,
            confidence_threshold=args.confidence_threshold
        )
    elif args.process_both:
        process_both_files(
            zdjecia_input=args.process_zdjecia,
            mieszkania_input=args.process_mieszkania,
            use_lora=args.use_lora,
            lora_weights=args.lora_weights,
            filter_interiors=not args.no_filter_interiors,
            confidence_threshold=args.confidence_threshold,
            batch_size=args.batch_size
        )
    elif args.process_zdjecia:
        process_zdjecia_txt_file(
            input_file=args.process_zdjecia,
            output_file=args.output_file_zdjecia,
            use_lora=args.use_lora,
            lora_weights=args.lora_weights,
            filter_interiors=not args.no_filter_interiors,
            confidence_threshold=args.confidence_threshold,
            batch_size=args.batch_size
        )
    elif args.process_mieszkania:
        process_mieszkania_json_file(
            mieszkania_file=args.process_mieszkania,
            output_file=args.output_file_mieszkania
        )
    else:
        print("Dostępne opcje:")
        print("  --process-zdjecia zdjecia.txt - przetwórz plik ze zdjęciami")
        print("  --process-mieszkania mieszkania.json - przetwórz plik z mieszkańami")
        print("  --process-both - przetwórz oba pliki sekwencyjnie")
        print("Dodatkowe opcje:")
        print("  --use-lora - użyj LoRA")
        print("  --lora-weights path - ścieżka do wag LoRA")
        print("  --no-filter-interiors - przetwarzaj wszystkie obrazy")
        print("  --confidence-threshold 0.3 - zmień próg pewności dla detekcji wnętrz")
        print("  --output-file-zdjecia - nazwa pliku wyjściowego dla zdjęć")
        print("  --output-file-mieszkania - nazwa pliku wyjściowego dla mieszkań")
        print("\nPrzykłady użycia:")
        print("  python main_v4_final.py --process-both")
        print("  python main_v4_final.py --process-zdjecia zdjecia.txt")
        print("  python main_v4_final.py --process-mieszkania mieszkania.json")