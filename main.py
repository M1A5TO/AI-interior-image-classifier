# main_v4.py
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

    # Convenience wrappers
    def analyze_image_from_url(self, url, filter_interiors=True):
        img = URLImageLoader.load_image_from_url(url)
        if img is None:
            return {'is_interior': False, 'reason': 'Failed to load image'}

        if filter_interiors:
            is_interior, confidence, category = self.detector.is_interior_image(img)
            if not is_interior:
                return {
                    'is_interior': False,
                    'interior_confidence': confidence,
                    'detected_category': category,
                    'analysis': {},
                    'reason': f'Not an interior image: {category}'
                }

        # Jeśli to wnętrze lub pomijamy filtrowanie - analizuj
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        analysis = self._analyze_image_tensor_fast(image_input)

        return {
            'is_interior': True,
            'interior_confidence': confidence if filter_interiors else 1.0,
            'detected_category': 'interior',
            'analysis': analysis,
            'reason': 'Success - interior image analyzed'
        }

    def _analyze_image_tensor_fast(self, image_input):
        results = {}
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for category, text_feats in self.text_features_cache.items():
                sims = (100.0 * image_features @ text_feats.T).softmax(dim=-1)
                vals, inds = sims[0].topk(min(5, text_feats.shape[0]))
                attrs = self.all_categories[category]
                results[category] = [(attrs[i], v.item()) for v, i in zip(vals, inds)]
        return results


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

    # Zapisz wyniki
    out_path = f"analysis_results_{len(images)}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n PODSUMOWANIE:")
    print(f"    Obrazy wnętrz: {interior_count}")
    print(f"    Obrazy nie-wnętrz: {non_interior_count}")
    print(f"    Wyniki zapisano do {out_path}")

    return out


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze-csv', type=str, help='csv with url column')
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--lora-weights', type=str, default='lora_models/comprehensive_lora.pth')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--no-filter-interiors', action='store_true',
                        help='Przetwarzaj wszystkie obrazy, nawet nie-wnętrza')
    parser.add_argument('--confidence-threshold', type=float, default=0.3, help='Próg pewności dla detekcji wnętrz')

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
    else:
        print("Run with --analyze-csv photos.csv [--use-lora --lora-weights path]")
        print("Dodatkowe opcje:")
        print("  --no-filter-interiors - przetwarzaj wszystkie obrazy")
        print("  --confidence-threshold 0.3 - zmień próg pewności dla detekcji wnętrz")