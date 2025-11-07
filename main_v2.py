import torch
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
from collections import Counter
import json
import os
import glob
import requests
from io import BytesIO
import pandas as pd
import csv


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        lora_result = (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)
        return lora_result


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRACLIPWrapper(nn.Module):
    def __init__(self, clip_model, rank=4, alpha=8):
        super().__init__()
        self.clip_model = clip_model
        self.rank = rank
        self.alpha = alpha
        self._replace_with_lora()

    def _replace_with_lora(self):
        """Poprawna aplikacja LoRA do modelu CLIP"""
        replaced_count = 0

        # Dla encoder-a tekstowego
        for name, module in self.clip_model.transformer.named_children():
            if isinstance(module, nn.Linear):
                # Zastąp bezpośrednie warstwy Linear
                lora_linear = LoRALinear(module, self.rank, self.alpha)
                setattr(self.clip_model.transformer, name, lora_linear)
                replaced_count += 1
                print(f"Zastąpiono warstwę: transformer.{name}")

        # Dla resblocks w transformerze
        if hasattr(self.clip_model.transformer, 'resblocks'):
            for idx, resblock in enumerate(self.clip_model.transformer.resblocks):
                # Zastąp warstwy w attention
                if hasattr(resblock.attn, 'in_proj_weight'):
                    # Pomijamy in_proj_weight - to jest inny typ warstwy
                    pass

                # Zastąp warstwy w MLP
                if hasattr(resblock, 'mlp'):
                    mlp = resblock.mlp
                    if hasattr(mlp, 'c_fc') and isinstance(mlp.c_fc, nn.Linear):
                        mlp.c_fc = LoRALinear(mlp.c_fc, self.rank, self.alpha)
                        replaced_count += 1
                        print(f"Zastąpiono warstwę: transformer.resblocks.{idx}.mlp.c_fc")

                    if hasattr(mlp, 'c_proj') and isinstance(mlp.c_proj, nn.Linear):
                        mlp.c_proj = LoRALinear(mlp.c_proj, self.rank, self.alpha)
                        replaced_count += 1
                        print(f"Zastąpiono warstwę: transformer.resblocks.{idx}.mlp.c_proj")

        print(f"Łącznie zastąpiono {replaced_count} warstw LoRA")

    def forward(self, *args, **kwargs):
        return self.clip_model(*args, **kwargs)

    def encode_image(self, image):
        return self.clip_model.encode_image(image)

    def encode_text(self, text):
        return self.clip_model.encode_text(text)


class InteriorStyleDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts, preprocess):
        self.image_paths = image_paths
        self.texts = texts
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.preprocess(image)
            text = clip.tokenize(self.texts[idx])
            return image, text
        except Exception as e:
            print(f"Błąd przy ładowaniu obrazu {self.image_paths[idx]}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            dummy_text = clip.tokenize("nowoczesne minimalistyczne wnętrze")
            return dummy_image, dummy_text


class URLImageLoader:
    """Klasa do ładowania obrazów z URL-i"""

    @staticmethod
    def load_image_from_url(url):
        """Ładuje obraz z URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Błąd przy ładowaniu obrazu z URL {url}: {e}")
            return None

    @staticmethod
    def load_images_from_csv(csv_path, max_images=None):
        """Ładuje URL-e obrazów z pliku CSV"""
        images_data = []
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                images_data.append({
                    'offer_id': row['offer_id'],
                    'seq': row['seq'],
                    'url': row['url']
                })

                if max_images and len(images_data) >= max_images:
                    break

            print(f"Załadowano {len(images_data)} URL-i obrazów z {csv_path}")
            return images_data
        except Exception as e:
            print(f"Błąd przy ładowaniu CSV: {e}")
            return []


class EnhancedLoRAInteriorAnalyzer:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Używanie urządzenia: {self.device}")

        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.lora_model = LoRACLIPWrapper(self.model, rank=4, alpha=8)
        self.lora_model.to(self.device)

        if model_path and os.path.exists(model_path):
            self.load_lora_weights(model_path)

        # Rozszerzone kategorie do treningu
        self.all_categories = self._load_all_categories()
        self.training_data = self._load_training_data_from_json()

    def _load_all_categories(self):
        """Ładuje wszystkie kategorie atrybutów z danych"""
        return {
            'styles': [
                "nowoczesny", "wielobarwny", "tematyczny - pojazdy", "rustykalny",
                "mid-century modern", "tradycyjny", "farmhouse", "indyjski",
                "nadmorski", "południowo-zachodni", "kobiecy", "eklektyczny",
                "azjatycki", "śródziemnomorski", "tropikalny", "boho chic",
                "współczesny", "minimalistyczny"
            ],
            'characteristics': self._extract_unique_attributes('characteristics'),
            'materials': self._extract_unique_attributes('materials'),
            'colors': self._extract_unique_attributes('colors'),
            'room_types': self._extract_unique_attributes('room_type')
        }

    def _extract_unique_attributes(self, attribute_type):
        """Ekstrahuje unikalne wartości atrybutów z danych treningowych"""
        unique_values = set()
        try:
            with open("interior_dataset.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data["training_data"]:
                if attribute_type in item:
                    if attribute_type == 'room_type':
                        unique_values.add(item[attribute_type])
                    else:
                        for value in item[attribute_type]:
                            unique_values.add(value)
        except Exception as e:
            print(f"Błąd przy ekstrakcji atrybutów {attribute_type}: {e}")

        return list(unique_values)

    def _load_training_data_from_json(self, json_path="interior_dataset.json"):
        """Ładuje dane treningowe z JSON"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["training_data"]
        except Exception as e:
            print(f"Błąd przy ładowaniu danych treningowych: {e}")
            return []

    def _generate_comprehensive_descriptions(self, item):
        """Generuje opisy treningowe dla WSZYSTKICH atrybutów"""
        descriptions = []

        # 1. Podstawowy styl
        descriptions.append(f"{item['style']} wnętrze")

        # 2. Style z różnymi kontekstami
        contexts = ["pokazuje", "przedstawia", "widać", "prezentuje"]
        for context in contexts:
            descriptions.append(f"zdjęcie {context} {item['style']} wnętrze")

        # 3. CHARAKTERYSTYKA - osobne opisy
        if 'characteristics' in item and item['characteristics']:
            for char in item['characteristics'][:3]:  # Weź 3 główne cechy
                descriptions.append(f"wnętrze z {char}")
                descriptions.append(f"{item['style']} {char}")
                descriptions.append(f"{char} w {item['room_type'] if 'room_type' in item else 'pomieszczeniu'}")

        # 4. MATERIAŁY - osobne opisy
        if 'materials' in item and item['materials']:
            for material in item['materials'][:3]:
                descriptions.append(f"wnętrze z {material}")
                descriptions.append(f"{item['style']} z materiałami {material}")
                descriptions.append(f"{material} w {item['room_type'] if 'room_type' in item else 'wnętrzu'}")

        # 5. KOLORY - osobne opisy
        if 'colors' in item and item['colors']:
            for color in item['colors'][:3]:
                descriptions.append(f"wnętrze w kolorze {color}")
                descriptions.append(f"{item['style']} {color}")
                descriptions.append(f"{color} {item['room_type'] if 'room_type' in item else 'pokój'}")

        # 6. TYP POKOJU - osobne opisy
        if 'room_type' in item and item['room_type']:
            descriptions.append(f"{item['room_type']}")
            descriptions.append(f"{item['room_type']} w stylu {item['style']}")
            descriptions.append(f"wnętrze {item['room_type']}")

        # 7. PEŁNE OPISY KOMBINUJĄCE WSZYSTKO
        full_description = f"{item['room_type']} w stylu {item['style']}"
        if 'characteristics' in item and item['characteristics']:
            full_description += f" z {', '.join(item['characteristics'][:2])}"
        if 'materials' in item and item['materials']:
            full_description += f" materiałami {', '.join(item['materials'][:2])}"
        if 'colors' in item and item['colors']:
            full_description += f" w kolorach {', '.join(item['colors'][:2])}"
        descriptions.append(full_description)

        return descriptions

    def load_training_data_from_json(self, json_path="interior_dataset.json"):
        """Ładuje dane treningowe z rozszerzonymi opisami"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            training_data = []

            for item in data["training_data"]:
                image_path = item["image_path"]

                # Generuj rozszerzone opisy dla wszystkich atrybutów
                descriptions = self._generate_comprehensive_descriptions(item)

                for desc in descriptions:
                    training_data.append((image_path, desc))

            print(f"Załadowano {len(training_data)} przykładów treningowych z {len(data['training_data'])} obrazów")
            return training_data

        except Exception as e:
            print(f"Błąd przy ładowaniu JSON: {e}")
            return []

    def fine_tune_comprehensive(self, json_path="interior_dataset.json", epochs=10,
                                learning_rate=5e-5, save_path="comprehensive_lora.pth"):
        """
        Trening obejmujący WSZYSTKIE atrybuty
        """
        print("Ładowanie rozszerzonych danych treningowych...")
        training_data = self.load_training_data_from_json(json_path)

        if not training_data:
            print("Brak danych treningowych! Koniec treningu.")
            return

        print(f"Rozpoczynanie treningu na {len(training_data)} przykładach...")

        # Przygotowanie danych
        image_paths = [data[0] for data in training_data]
        texts = [data[1] for data in training_data]

        dataset = InteriorStyleDataset(image_paths, texts, self.preprocess)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # Tylko parametry LoRA są trenowane
        trainable_params = []
        for name, param in self.lora_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        print(f"Liczba trenowanych parametrów LoRA: {len(trainable_params)}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.98)
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        criterion = torch.nn.CrossEntropyLoss()

        self.lora_model.train()

        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0

            for batch_idx, (images, texts) in enumerate(dataloader):
                try:
                    images = images.to(self.device)
                    texts = texts.squeeze(1).to(self.device)

                    # Forward pass
                    logits_per_image, logits_per_text = self.lora_model(images, texts)

                    # Contrastive loss
                    batch_size = images.size(0)
                    labels = torch.arange(batch_size).to(self.device)

                    loss = (criterion(logits_per_image, labels) + criterion(logits_per_text, labels)) / 2

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1

                    if batch_idx % 20 == 0:
                        print(f"Epoka {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

                except Exception as e:
                    print(f"Błąd w batch {batch_idx}: {e}")
                    continue

            scheduler.step()

            if batch_count > 0:
                avg_loss = total_loss / batch_count
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoka {epoch + 1}/{epochs}, Średni Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        # Zapisz wagi LoRA
        self.save_lora_weights(save_path)
        print(f"Zapisano wagi LoRA do: {save_path}")

    def analyze_image_from_url(self, image_url):
        """Analizuje obraz z URL"""
        try:
            image = URLImageLoader.load_image_from_url(image_url)
            if image is None:
                print(f"Nie udało się załadować obrazu z URL: {image_url}")
                return {}

            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            return self._analyze_image_tensor(image_input)
        except Exception as e:
            print(f"Błąd podczas analizy obrazu z URL {image_url}: {e}")
            return {}

    def analyze_all_attributes(self, image_path):
        """Analizuje WSZYSTKIE atrybuty bezpośrednio przez model"""
        try:
            if image_path.startswith('http'):
                return self.analyze_image_from_url(image_path)

            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            return self._analyze_image_tensor(image_input)

        except Exception as e:
            print(f"Błąd podczas analizy: {e}")
            return {}

    def _analyze_image_tensor(self, image_input):
        """Analizuje tensor obrazu"""
        results = {}

        with torch.no_grad():
            # Analiza dla każdej kategorii
            for category, attributes in self.all_categories.items():
                if not attributes:  # Pomijaj puste kategorie
                    continue

                print(f"Analizowanie {category}...")

                category_results = []

                # Różne konteksty dla lepszej generalizacji
                contexts = ["", "zdjęcie pokazuje ", "na obrazie widać "]

                for context in contexts:
                    # Przygotuj teksty dla tej kategorii
                    if category == 'room_types':
                        text_inputs = torch.cat([
                            clip.tokenize(f"{context}{attr}") for attr in attributes
                        ]).to(self.device)
                    else:
                        text_inputs = torch.cat([
                            clip.tokenize(f"{context}wnętrze z {attr}") for attr in attributes
                        ]).to(self.device)

                    # Enkoduj obraz i teksty
                    image_features = self.lora_model.encode_image(image_input)
                    text_features = self.lora_model.encode_text(text_inputs)

                    # Normalizuj cechy
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Oblicz podobieństwo
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(min(5, len(attributes)))

                    for value, idx in zip(values, indices):
                        category_results.append((attributes[idx], value.item()))

                # Agreguj wyniki z różnych kontekstów
                aggregated = self._aggregate_category_results(category_results)
                results[category] = aggregated

        return results

    def _aggregate_category_results(self, results):
        """Agreguje wyniki dla pojedynczej kategorii"""
        if not results:
            return []

        counter = Counter()
        for attribute, score in results:
            counter[attribute] += score

        total = sum(counter.values())
        final_results = [(attr, (score / total) * 100) for attr, score in counter.most_common(5)]
        return final_results

    def generate_comprehensive_report(self, image_path):
        """Generuje kompletny raport ze wszystkimi atrybutami"""
        results = self.analyze_all_attributes(image_path)

        print("=" * 70)
        print("KOMPLETNA ANALIZA WNĘTRZA")
        print("=" * 70)

        if not results:
            print("Nie udało się przeanalizować obrazu.")
            return

        # Wyświetl wyniki dla każdej kategorii
        categories_order = ['room_types', 'styles', 'characteristics', 'materials', 'colors']

        for category in categories_order:
            if category in results and results[category]:
                category_name = category.upper().replace('_', ' ')
                print(f"\n{category_name}:")
                for i, (attr, confidence) in enumerate(results[category][:5], 1):
                    print(f"  {i}. {attr}: {confidence:.2f}%")

        # Analiza techniczna
        self._technical_analysis(image_path)

        return results

    def save_lora_weights(self, path):
        lora_weights = {}
        for name, param in self.lora_model.named_parameters():
            if 'lora' in name:
                lora_weights[name] = param.data.cpu()
        torch.save(lora_weights, path)

    def load_lora_weights(self, path):
        lora_weights = torch.load(path, map_location=self.device)
        for name, param in self.lora_model.named_parameters():
            if name in lora_weights:
                param.data = lora_weights[name]
        print(f"Załadowano wagi LoRA z: {path}")

    def _technical_analysis(self, image_path):
        try:
            if image_path.startswith('http'):
                image = URLImageLoader.load_image_from_url(image_path)
                if image is None:
                    return
            else:
                image = Image.open(image_path)

            img_array = np.array(image)

            hsv_img = image.convert('HSV')
            hsv_array = np.array(hsv_img)

            brightness = np.mean(img_array) / 255.0 * 100
            saturation = np.mean(hsv_array[:, :, 1]) / 255.0 * 100
            color_variance = np.std(img_array, axis=(0, 1)).mean()

            print(f"\nANALIZA TECHNICZNA:")
            print(f"• Średnia jasność: {brightness:.1f}%")
            print(f"• Nasycenie kolorów: {saturation:.1f}%")
            print(f"• Różnorodność kolorów: {color_variance:.1f}")
        except Exception as e:
            print(f"\nBłąd podczas analizy technicznej: {e}")


class SimpleInteriorAnalyzer:
    """Uproszczona wersja do szybkiego użycia bez treningu"""

    def __init__(self, use_lora=False, lora_weights_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)

        # Załaduj dane treningowe dla atrybutów
        self.training_data = self._load_training_data()
        self.all_categories = self._extract_all_categories()

        if use_lora and lora_weights_path and os.path.exists(lora_weights_path):
            print("Ładowanie wag LoRA...")
            self.lora_model = LoRACLIPWrapper(self.model, rank=4, alpha=8)
            self.lora_model.to(self.device)

            lora_weights = torch.load(lora_weights_path, map_location=self.device)
            for name, param in self.lora_model.named_parameters():
                if name in lora_weights:
                    param.data = lora_weights[name]
            self.model = self.lora_model
            print("Używam modelu z LoRA")
        else:
            print("Używanie standardowego modelu CLIP")

    def _load_training_data(self, json_path="interior_dataset.json"):
        """Ładuje dane treningowe z JSON"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["training_data"]
        except Exception as e:
            print(f"Błąd przy ładowaniu danych treningowych: {e}")
            return []

    def _extract_all_categories(self):
        """Ekstrahuje wszystkie kategorie atrybutów"""
        styles = set()
        characteristics = set()
        materials = set()
        colors = set()
        room_types = set()

        for item in self.training_data:
            styles.add(item['style'])
            room_types.add(item['room_type'])

            for char in item.get('characteristics', []):
                characteristics.add(char)
            for material in item.get('materials', []):
                materials.add(material)
            for color in item.get('colors', []):
                colors.add(color)

        return {
            'styles': list(styles),
            'characteristics': list(characteristics),
            'materials': list(materials),
            'colors': list(colors),
            'room_types': list(room_types)
        }

    def analyze_image(self, image_path):
        """Główna funkcja analizy obrazu"""
        return self.generate_comprehensive_report(image_path)

    def analyze_image_from_url(self, image_url):
        """Analizuje obraz z URL"""
        try:
            image = URLImageLoader.load_image_from_url(image_url)
            if image is None:
                print(f"Nie udało się załadować obrazu z URL: {image_url}")
                return {}

            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            return self._analyze_image_tensor(image_input)
        except Exception as e:
            print(f"Błąd podczas analizy obrazu z URL {image_url}: {e}")
            return {}

    def analyze_all_attributes(self, image_path):
        """Analizuje wszystkie atrybuty"""
        try:
            if image_path.startswith('http'):
                return self.analyze_image_from_url(image_path)

            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            return self._analyze_image_tensor(image_input)

        except Exception as e:
            print(f"Błąd podczas analizy: {e}")
            return {}

    def _analyze_image_tensor(self, image_input):
        """Analizuje tensor obrazu"""
        results = {}

        with torch.no_grad():
            for category, attributes in self.all_categories.items():
                if not attributes:
                    continue

                print(f"Analizowanie {category}...")
                category_results = []

                contexts = ["", "zdjęcie pokazuje ", "na obrazie widać "]

                for context in contexts:
                    if category == 'room_types':
                        text_inputs = torch.cat([
                            clip.tokenize(f"{context}{attr}") for attr in attributes
                        ]).to(self.device)
                    else:
                        text_inputs = torch.cat([
                            clip.tokenize(f"{context}wnętrze z {attr}") for attr in attributes
                        ]).to(self.device)

                    image_features = self.model.encode_image(image_input)
                    text_features = self.model.encode_text(text_inputs)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(min(5, len(attributes)))

                    for value, idx in zip(values, indices):
                        category_results.append((attributes[idx], value.item()))

                # Agreguj wyniki
                counter = Counter()
                for attr, score in category_results:
                    counter[attr] += score
                total = sum(counter.values())
                final_results = [(attr, (score / total) * 100) for attr, score in counter.most_common(5)]
                results[category] = final_results

        return results

    def generate_comprehensive_report(self, image_path):
        """Generuje kompletny raport"""
        results = self.analyze_all_attributes(image_path)

        print("=" * 70)
        print("ANALIZA WNĘTRZA")
        print("=" * 70)

        if not results:
            print("Nie udało się przeanalizować obrazu.")
            return

        # Wyświetl wyniki
        categories_order = ['room_types', 'styles', 'characteristics', 'materials', 'colors']

        for category in categories_order:
            if category in results and results[category]:
                category_name = category.upper().replace('_', ' ')
                print(f"\n{category_name}:")
                for i, (attr, confidence) in enumerate(results[category][:5], 1):
                    print(f"  {i}. {attr}: {confidence:.2f}%")

        # Analiza techniczna
        self._technical_analysis(image_path)

        return results

    def _technical_analysis(self, image_path):
        try:
            if image_path.startswith('http'):
                image = URLImageLoader.load_image_from_url(image_path)
                if image is None:
                    return
            else:
                image = Image.open(image_path)

            img_array = np.array(image)

            hsv_img = image.convert('HSV')
            hsv_array = np.array(hsv_img)

            brightness = np.mean(img_array) / 255.0 * 100
            saturation = np.mean(hsv_array[:, :, 1]) / 255.0 * 100
            color_variance = np.std(img_array, axis=(0, 1)).mean()

            print(f"\nANALIZA TECHNICZNA:")
            print(f"• Średnia jasność: {brightness:.1f}%")
            print(f"• Nasycenie kolorów: {saturation:.1f}%")
            print(f"• Różnorodność kolorów: {color_variance:.1f}")
        except Exception as e:
            print(f"\nBłąd podczas analizy technicznej: {e}")


def analyze_images_from_csv(csv_path, use_lora=False, lora_weights_path=None, max_images=None):
    """Analizuje wszystkie obrazy z pliku CSV"""
    print(f"Ładowanie URL-i obrazów z {csv_path}...")
    images_data = URLImageLoader.load_images_from_csv(csv_path, max_images)

    if not images_data:
        print("Nie udało się załadować żadnych URL-i obrazów.")
        return

    if use_lora:
        print("ANALIZA Z WYTRENOWANYM MODELEM LoRA...")
        analyzer = SimpleInteriorAnalyzer(use_lora=True, lora_weights_path=lora_weights_path)
    else:
        print("ANALIZA Z STANDARDOWYM MODELEM CLIP...")
        analyzer = SimpleInteriorAnalyzer(use_lora=False)

    results = {}

    for i, image_data in enumerate(images_data):
        print(f"\n{'=' * 80}")
        print(f"ANALIZA OBRAZU {i + 1}/{len(images_data)}")
        print(f"Offer ID: {image_data['offer_id']}, Seq: {image_data['seq']}")
        print(f"URL: {image_data['url']}")
        print(f"{'=' * 80}")

        try:
            image_result = analyzer.analyze_image(image_data['url'])
            results[f"{image_data['offer_id']}_{image_data['seq']}"] = {
                'url': image_data['url'],
                'analysis': image_result
            }
        except Exception as e:
            print(f"Błąd podczas analizy obrazu {image_data['url']}: {e}")
            results[f"{image_data['offer_id']}_{image_data['seq']}"] = {
                'url': image_data['url'],
                'error': str(e)
            }

    # Zapisz wyniki do pliku
    output_file = f"analysis_results_{len(images_data)}_images.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nWyniki zapisano do: {output_file}")
    return results


# GŁÓWNE UŻYCIE
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analiza stylów wnętrz')
    parser.add_argument('--train', action='store_true', help='Przeprowadź trening LoRA')
    parser.add_argument('--analyze', type=str, help='Ścieżka do obrazu do analizy')
    parser.add_argument('--analyze-csv', type=str, help='Ścieżka do pliku CSV z URL-ami obrazów')
    parser.add_argument('--use-lora', action='store_true', help='Użyj wytrenowanego modelu LoRA')
    parser.add_argument('--lora-weights', type=str, default='lora_models/comprehensive_lora.pth',
                        help='Ścieżka do wag LoRA')
    parser.add_argument('--max-images', type=int, help='Maksymalna liczba obrazów do analizy z CSV')

    args = parser.parse_args()

    if args.train:
        # TRENING WSZYSTKICH ATRYBUTÓW
        print("ROZPOCZĘCIE TRENINGU...")
        analyzer = EnhancedLoRAInteriorAnalyzer()
        analyzer.fine_tune_comprehensive(
            json_path="interior_dataset.json",
            epochs=10,
            learning_rate=5e-5,
            save_path=args.lora_weights
        )

    elif args.analyze_csv:
        # ANALIZA WSZYSTKICH OBRAZÓW Z CSV
        analyze_images_from_csv(
            csv_path=args.analyze_csv,
            use_lora=args.use_lora,
            lora_weights_path=args.lora_weights,
            max_images=args.max_images
        )

    elif args.analyze:
        # ANALIZA POJEDYNCZEGO OBRAZU
        if args.use_lora:
            print("ANALIZA Z WYTRENOWANYM MODELEM LoRA...")
            analyzer = SimpleInteriorAnalyzer(use_lora=True, lora_weights_path=args.lora_weights)
        else:
            print("ANALIZA Z STANDARDOWYM MODELEM CLIP...")
            analyzer = SimpleInteriorAnalyzer(use_lora=False)

        if args.analyze.startswith('http'):
            print(f"Analizowanie obrazu z URL: {args.analyze}")
        else:
            print(f"Analizowanie obrazu: {args.analyze}")

        results = analyzer.analyze_image(args.analyze)

    else:
        # PRZYKŁAD UŻYCIA
        print("Przykład użycia:")
        print("python main.py --train  # Trening modelu")
        print("python main.py --analyze obraz.jpg --use-lora  # Analiza z LoRA")
        print("python main.py --analyze obraz.jpg  # Analiza bez LoRA")
        print("python main.py --analyze-csv photos.csv --use-lora  # Analiza wszystkich obrazów z CSV")
        print("python main.py --analyze-csv photos.csv --max-images 10  # Analiza tylko 10 obrazów z CSV")