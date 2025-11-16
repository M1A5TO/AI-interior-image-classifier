import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
import json
import os
import argparse
from tqdm import tqdm


# ------------------------
# Ulepszone LoRA modules - POPRAWIONE
# ------------------------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        # Inicjalizacja parametrów LoRA
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return self.dropout(lora_output)


class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features,
                              rank, alpha, dropout)

        # Zachowaj oryginalne właściwości
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRACLIPWrapper(nn.Module):
    """Wrapper CLIP z LoRA - POPRAWIONA REPLACJA"""

    def __init__(self, clip_model, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.clip_model = clip_model
        self.rank = rank
        self.alpha = alpha

        # Zastosuj LoRA do text encoder
        self._replace_text_linears_with_lora()

        # DEBUG: wypisz parametry LoRA
        self._print_lora_parameters()

    def _replace_text_linears_with_lora(self):
        """Zastąp warstwy Linear w text encoderze LoRALinear"""
        replaced_count = 0

        # Przejdź przez wszystkie moduły w transformerze tekstu
        for name, module in self.clip_model.transformer.named_children():
            if isinstance(module, nn.Linear):
                # Zamień na LoRALinear
                new_module = LoRALinear(module, self.rank, self.alpha)
                setattr(self.clip_model.transformer, name, new_module)
                replaced_count += 1
                print(f"✅ Zastąpiono: transformer.{name}")

        # Przejdź przez resblocks w transformerze
        if hasattr(self.clip_model.transformer, 'resblocks'):
            for idx, resblock in enumerate(self.clip_model.transformer.resblocks):
                # Sprawdź attention
                if hasattr(resblock, 'attn'):
                    attn = resblock.attn
                    if hasattr(attn, 'out_proj') and isinstance(attn.out_proj, nn.Linear):
                        attn.out_proj = LoRALinear(attn.out_proj, self.rank, self.alpha)
                        replaced_count += 1
                        print(f"✅ Zastąpiono: transformer.resblocks.{idx}.attn.out_proj")

                # Sprawdź MLP
                if hasattr(resblock, 'mlp'):
                    mlp = resblock.mlp
                    # c_fc (linear 1)
                    if hasattr(mlp, 'c_fc') and isinstance(mlp.c_fc, nn.Linear):
                        mlp.c_fc = LoRALinear(mlp.c_fc, self.rank, self.alpha)
                        replaced_count += 1
                        print(f"✅ Zastąpiono: transformer.resblocks.{idx}.mlp.c_fc")
                    # c_proj (linear 2)
                    if hasattr(mlp, 'c_proj') and isinstance(mlp.c_proj, nn.Linear):
                        mlp.c_proj = LoRALinear(mlp.c_proj, self.rank, self.alpha)
                        replaced_count += 1
                        print(f"✅ Zastąpiono: transformer.resblocks.{idx}.mlp.c_proj")

        print(f"🎯 Łącznie zastąpiono {replaced_count} warstw Linear → LoRALinear")

    def _print_lora_parameters(self):
        """Wypisz wszystkie parametry LoRA dla debugowania"""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora' in name:
                lora_params.append((name, param.shape, param.requires_grad))

        print(f"🔍 Znaleziono {len(lora_params)} parametrów LoRA:")
        for name, shape, requires_grad in lora_params:
            print(f"   {name}: {shape}, trainable: {requires_grad}")

    def forward(self, images=None, text=None):
        """Forward pass - deleguje do oryginalnego CLIP"""
        return self.clip_model(images, text)


# ------------------------
# Dataset (bez zmian)
# ------------------------
class InteriorStyleDataset(Dataset):
    def __init__(self, json_path, preprocess):
        self.data = []
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)["training_data"]

        for item in raw_data:
            img_path = item["image_path"]
            # Wiele promptów tekstowych dla lepszego uczenia
            styles = [f"{item['style']} wnętrze"]
            if item.get('room_type'):
                styles.append(f"{item['room_type']} w stylu {item['style']}")
            if item.get('characteristics'):
                for char in item['characteristics'][:2]:
                    styles.append(f"{char} {item['style']} wnętrze")

            self.data.append((img_path, styles))

        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, texts = self.data[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image)

            # Losowy wybór promptu tekstowego
            text = texts[torch.randint(0, len(texts), (1,)).item()]
            text_tokens = clip.tokenize([text])[0]

            return image, text_tokens
        except Exception as e:
            print(f"Błąd ładowania {img_path}: {e}")
            # Fallback
            dummy_image = torch.zeros(3, 224, 224)
            dummy_text = clip.tokenize(["wnętrze"])[0]
            return dummy_image, dummy_text


# ------------------------
# Ulepszony trening - POPRAWIONE
# ------------------------
def train_lora(json_path, save_path="lora_models/comprehensive_lora_improved.pth",
               epochs=20, batch_size=8, lr=1e-4, rank=16, alpha=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Używam urządzenia: {device}")

    # Załaduj model CLIP
    print("📥 Ładowanie modelu CLIP...")
    model, preprocess = clip.load("ViT-B/16", device=device)  # Używamy ViT-B/16 dla kompatybilności

    # Zastosuj LoRA
    print("🔧 Stosowanie LoRA do modelu...")
    lora_wrapper = LoRACLIPWrapper(model, rank=rank, alpha=alpha)
    lora_wrapper.to(device)

    # Dataset i DataLoader
    print("📊 Przygotowanie danych...")
    dataset = InteriorStyleDataset(json_path, preprocess)

    # Podział na train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ZNAJDŹ PARAMETRY DO TRENOWANIA - POPRAWIONE
    trainable_params = []
    for name, param in lora_wrapper.named_parameters():
        if 'lora' in name and param.requires_grad:
            trainable_params.append(param)
            print(f"🎯 Parametr do trenowania: {name}, kształt: {param.shape}")

    if len(trainable_params) == 0:
        print("❌ BŁAD: Nie znaleziono żadnych parametrów do trenowania!")
        print("Dostępne parametry:")
        for name, param in lora_wrapper.named_parameters():
            print(f"   {name}: {param.shape}, requires_grad: {param.requires_grad}")
        return

    print(f"🎯 Liczba trenowanych parametrów LoRA: {len(trainable_params):,}")
    total_params = sum(p.numel() for p in trainable_params)
    print(f"🎯 Łączna liczba parametrów LoRA: {total_params:,}")

    # Optimizer i loss
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    print("🚀 Rozpoczynanie treningu...")
    for epoch in range(epochs):
        # TRENING
        lora_wrapper.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')

        for batch_idx, (images, texts) in enumerate(train_bar):
            images = images.to(device)
            texts = texts.to(device)

            # Forward pass
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Text features przechodzą przez LoRA
            text_features = lora_wrapper.clip_model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Oblicz podobieństwo
            logit_scale = model.logit_scale.exp()
            logits_per_image = (image_features @ text_features.t()) * logit_scale
            logits_per_text = logits_per_image.t()

            labels = torch.arange(images.size(0), device=device)
            loss = (criterion(logits_per_image, labels) + criterion(logits_per_text, labels)) / 2

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # WALIDACJA
        lora_wrapper.eval()
        val_loss = 0
        with torch.no_grad():
            for images, texts in val_loader:
                images = images.to(device)
                texts = texts.to(device)

                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                text_features = lora_wrapper.clip_model.encode_text(texts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits_per_image = (image_features @ text_features.t()) * model.logit_scale.exp()
                logits_per_text = logits_per_image.t()

                labels = torch.arange(images.size(0), device=device)
                loss = (criterion(logits_per_image, labels) + criterion(logits_per_text, labels)) / 2
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        print(f"📊 Epoch {epoch + 1}/{epochs}:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Zapisz najlepszy model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Zapisz TYLKO parametry LoRA
            lora_state_dict = {}
            for name, param in lora_wrapper.named_parameters():
                if 'lora' in name:
                    lora_state_dict[name] = param.data.cpu()

            torch.save(lora_state_dict, save_path)
            print(f"   💾 Zapisano najlepszy model (val_loss: {avg_val_loss:.4f})")

    print(f"✅ Trening zakończony. Najlepszy val_loss: {best_val_loss:.4f}")
    return best_val_loss


# ------------------------
# Funkcja testująca
# ------------------------
def test_trained_lora(json_path, lora_weights_path):
    """Test wytrenowanego modelu LoRA"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    # Załaduj LoRA
    lora_wrapper = LoRACLIPWrapper(model)
    lora_wrapper.to(device)

    # Załaduj wagi LoRA
    if os.path.exists(lora_weights_path):
        lora_weights = torch.load(lora_weights_path, map_location='cpu')
        # Załaduj wagi do modelu
        lora_state_dict = {}
        for name, param in lora_wrapper.named_parameters():
            if 'lora' in name and name in lora_weights:
                lora_state_dict[name] = lora_weights[name]

        lora_wrapper.load_state_dict(lora_state_dict, strict=False)
        print(f"✅ Załadowano wagi LoRA z: {lora_weights_path}")
    else:
        print(f"❌ Plik wag nie istnieje: {lora_weights_path}")
        return

    # Test
    dataset = InteriorStyleDataset(json_path, preprocess)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, texts in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            texts = texts.to(device)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = lora_wrapper.clip_model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.t()) * model.logit_scale.exp()
            predictions = similarities.argmax(dim=1)
            labels = torch.arange(images.size(0), device=device)

            correct += (predictions == labels).sum().item()
            total += images.size(0)

    accuracy = correct / total
    print(f"🎯 Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, default='interior_dataset.json')
    parser.add_argument('--save-path', type=str, default='lora_models/comprehensive_lora_improved.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=32)
    parser.add_argument('--test', action='store_true', help='Test trained model')

    args = parser.parse_args()

    if args.test:
        test_trained_lora(args.json_path, args.save_path)
    else:
        train_lora(
            json_path=args.json_path,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            rank=args.rank,
            alpha=args.alpha
        )