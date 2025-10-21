import torch
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
from collections import Counter
import json
import os
import glob


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
        for name, module in self.clip_model.transformer.named_children():
            if isinstance(module, nn.Linear):
                lora_linear = LoRALinear(module, self.rank, self.alpha)
                setattr(self.clip_model.transformer, name, lora_linear)
            elif isinstance(module, nn.ModuleList):
                for block_idx, block in enumerate(module):
                    self._replace_in_block(block, f"resblocks.{block_idx}")

    def _replace_in_block(self, block, block_path):
        for name, module in block.named_children():
            full_name = f"{block_path}.{name}"
            if isinstance(module, nn.Linear):
                lora_linear = LoRALinear(module, self.rank, self.alpha)
                setattr(block, name, lora_linear)
            elif isinstance(module, nn.Module):
                self._replace_in_block(module, full_name)

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


class JSONInteriorAnalyzer:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Używanie urządzenia: {self.device}")

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.lora_model = LoRACLIPWrapper(self.model, rank=4, alpha=8)
        self.lora_model.to(self.device)

        if model_path and os.path.exists(model_path):
            self.load_lora_weights(model_path)

        self.interior_styles = self._load_interior_styles()

    def _load_interior_styles(self):
        return [
            "nowoczesne minimalistyczne wnętrze",
            "styl skandynawski z jasnym drewnem",
            "industrialny loft z cegłą",
            "rustykalne wnętrze z naturalnymi materiałami",
            "styl boho z tekstyliami i roślinami",
            "tradycyjne klasyczne wnętrze",
            "glamour z lustrami i błyszczącymi powierzchniami",
        ]

    def load_training_data_from_json(self, json_path="interior_dataset.json"):
        """
        Ładuje dane treningowe z pliku JSON
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            training_data = []

            for item in data["training_data"]:
                image_path = item["image_path"]

                # Generuj różne opisy na podstawie danych JSON
                descriptions = self._generate_descriptions(item)

                for desc in descriptions:
                    training_data.append((image_path, desc))

            print(f"Załadowano {len(training_data)} przykładów treningowych z {len(data['training_data'])} obrazów")
            return training_data

        except FileNotFoundError:
            print(f"Plik {json_path} nie istnieje!")
            return []
        except Exception as e:
            print(f"Błąd przy ładowaniu JSON: {e}")
            return []

    def _generate_descriptions(self, item):
        """
        Generuje różne warianty opisów z danych JSON
        """
        descriptions = []

        # Podstawowy styl
        descriptions.append(f"{item['style']} wnętrze")

        # Styl + główne cechy
        if 'characteristics' in item and item['characteristics']:
            main_chars = ', '.join(item['characteristics'][:3])
            descriptions.append(f"{item['style']} wnętrze z {main_chars}")

        # Styl + materiały
        if 'materials' in item and item['materials']:
            materials = ', '.join(item['materials'][:3])
            descriptions.append(f"{item['style']} wnętrze z materiałami {materials}")

        # Styl + kolory
        if 'colors' in item and item['colors']:
            colors = ', '.join(item['colors'][:3])
            descriptions.append(f"{item['style']} wnętrze w kolorach {colors}")

        # Typ pomieszczenia + styl
        if 'room_type' in item and item['room_type']:
            descriptions.append(f"{item['room_type']} w stylu {item['style']}")

        # Pełny opis
        full_parts = []
        if 'room_type' in item and item['room_type']:
            full_parts.append(item['room_type'])

        full_parts.append(f"styl {item['style']}")

        if 'characteristics' in item and item['characteristics']:
            full_parts.append(f"cechy: {', '.join(item['characteristics'][:2])}")

        if 'materials' in item and item['materials']:
            full_parts.append(f"materiały: {', '.join(item['materials'][:2])}")

        descriptions.append(' '.join(full_parts))

        return descriptions

    def fine_tune(self, json_path="interior_dataset.json", epochs=5, learning_rate=1e-4, save_path="trained_lora.pth"):
        """
        Trening na podstawie danych z JSON
        """
        print("Ładowanie danych treningowych z JSON...")
        training_data = self.load_training_data_from_json(json_path)

        if not training_data:
            print("Brak danych treningowych! Koniec treningu.")
            return

        print(f"Rozpoczynanie treningu na {len(training_data)} przykładach...")

        # Przygotowanie danych
        image_paths = [data[0] for data in training_data]
        texts = [data[1] for data in training_data]

        dataset = InteriorStyleDataset(image_paths, texts, self.preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

        # Tylko parametry LoRA są trenowane
        trainable_params = []
        for name, param in self.lora_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        print(f"Liczba trenowanych parametrów LoRA: {len(trainable_params)}")

        if not trainable_params:
            print("Brak parametrów LoRA do trenowania!")
            return

        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
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
                    optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1

                    if batch_idx % 10 == 0:
                        print(f"Epoka {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

                except Exception as e:
                    print(f"Błąd w batch {batch_idx}: {e}")
                    continue

            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print(f"Epoka {epoch + 1}/{epochs}, Średni Loss: {avg_loss:.4f}")

        # Zapisz wagi LoRA
        self.save_lora_weights(save_path)
        print(f"Zapisano wagi LoRA do: {save_path}")

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

    def analyze_interior_style(self, image_path):
        try:
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            contexts = [
                "zdjęcie pokazuje ",
                "na obrazie widać ",
                "fotografia przedstawia ",
            ]

            all_results = []

            with torch.no_grad():
                for context in contexts:
                    text_inputs = torch.cat([
                        clip.tokenize(context + style) for style in self.interior_styles
                    ]).to(self.device)

                    image_features = self.lora_model.encode_image(image_input)
                    text_features = self.lora_model.encode_text(text_inputs)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(5)

                    for value, idx in zip(values, indices):
                        all_results.append((self.interior_styles[idx], value.item()))

            # Agregacja wyników
            aggregated = self._aggregate_results(all_results)
            return aggregated
        except Exception as e:
            print(f"Błąd podczas analizy: {e}")
            return []

    def _aggregate_results(self, results):
        if not results:
            return []

        counter = Counter()
        for style, score in results:
            key_words = style.split()[:3]
            key = ' '.join(key_words)
            counter[key] += score

        total = sum(counter.values())
        final_results = [(style, (score / total) * 100) for style, score in counter.most_common(6)]
        return final_results

    def generate_detailed_style_report(self, image_path):
        results = self.analyze_interior_style(image_path)

        print("=" * 60)
        print("ANALIZA STYLU WNĘTRZA (z LoRA)")
        print("=" * 60)

        if not results:
            print("Nie udało się przeanalizować obrazu.")
            return

        for i, (style, confidence) in enumerate(results, 1):
            print(f"{i}. {style}: {confidence:.2f}%")

        if results:
            dominant_style, dominant_conf = results[0]
            print(f"\nDOMINUJĄCY STYL: {dominant_style} ({dominant_conf:.1f}%)")

        self._technical_analysis(image_path)

    def _technical_analysis(self, image_path):
        try:
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


# Uproszczona klasa do użycia
class InteriorAnalyzer:
    def __init__(self, use_lora=False, lora_weights_path="trained_lora.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

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
            print("Używanie standardowego modelu CLiP")

        self.interior_styles = [
            "nowoczesne minimalistyczne wnętrze",
            "styl skandynawski z jasnym drewnem",
            "industrialny loft z cegłą",
            "rustykalne wnętrze z naturalnymi materiałami",
            "styl boho z tekstyliami i roślinami",
            "tradycyjne klasyczne wnętrze",
            "glamour z lustrami i błyszczącymi powierzchniami",
        ]

    def analyze(self, image_path):
        return self.generate_detailed_style_report(image_path)

    def analyze_interior_style(self, image_path):
        try:
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            contexts = ["zdjęcie pokazuje ", "na obrazie widać "]

            all_results = []

            with torch.no_grad():
                for context in contexts:
                    text_inputs = torch.cat([
                        clip.tokenize(context + style) for style in self.interior_styles
                    ]).to(self.device)

                    image_features = self.model.encode_image(image_input)
                    text_features = self.model.encode_text(text_inputs)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(5)

                    for value, idx in zip(values, indices):
                        all_results.append((self.interior_styles[idx], value.item()))

            counter = Counter()
            for style, score in all_results:
                key_words = style.split()[:3]
                key = ' '.join(key_words)
                counter[key] += score

            total = sum(counter.values())
            final_results = [(style, (score / total) * 100) for style, score in counter.most_common(6)]
            return final_results
        except Exception as e:
            print(f"Błąd podczas analizy: {e}")
            return []

    def generate_detailed_style_report(self, image_path):
        results = self.analyze_interior_style(image_path)

        print("=" * 60)
        print("ANALIZA STYLU WNĘTRZA")
        print("=" * 60)

        if not results:
            print("Nie udało się przeanalizować obrazu.")
            return

        for i, (style, confidence) in enumerate(results, 1):
            print(f"{i}. {style}: {confidence:.2f}%")

        if results:
            dominant_style, dominant_conf = results[0]
            print(f"\nDOMINUJĄCY STYL: {dominant_style} ({dominant_conf:.1f}%)")

        self._technical_analysis(image_path)

    def _technical_analysis(self, image_path):
        try:
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


# GŁÓWNE UŻYCIE
if __name__ == "__main__":
    # OPCJA 1: TRENING Z JSON
    # analyzer = JSONInteriorAnalyzer()
    # analyzer.fine_tune(json_path="interior_dataset.json", epochs=3, save_path="trained_lora.pth")

    # OPCJA 2: UŻYCIE WYTRENOWANEGO MODELU
    analyzer = InteriorAnalyzer(use_lora=False)  # Zmień na True po treningu
    analyzer.analyze("interior_sample.jpg")