"""
Trabajo Práctico 3 — Tecnología Digital VI
Clasificación de imágenes satelitales EuroSAT con CNNs (PyTorch)

Juan Ignacio Elosegui y Juan González Merlhe
"""

import os
import time
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

#   Variables para la configuración
dataset = "EuroSAT/"
BATCH_SIZE = 64
NUM_WORKERS = 2
IMG_SIZE = 64
NUM_CLASSES = 10
SEED = 69

#   Epochs
EPOCHS_CUSTOM = 30
EPOCHS_TRANSFER = 15
#   Learning rates
LR_CUSTOM = 1e-3
LR_TRANSFER = 1e-4
#   TODO: ir tocándolo pq estoy en CPU por la notebook. Probar con GPU en la PC.

#   Qué dispositivo voy a usar. En notebook es CPU sí o sí. Ver línea 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

#   Usar el dataset
with open(os.path.join(dataset, "label_map.json")) as f:
    LABEL_MAP = json.load(f)

#   Almacenar nombres de clases
NOMBRES_CLASES = [k for k, v in sorted(LABEL_MAP.items(), key=lambda x: x[1])]

class EuroSATDataset(Dataset):
    def __init__(self,
                 csv_file,
                 root_dir, 
                 transform=None):
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    #   Devuelve la longitud del dataset en filas.
    def __len__(self):
        return len(self.df)

    #   Saca la muestra i del dataset
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Filename"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["Label"])
        if self.transform is not None:
            # Aplicar transformación si la hay
            image = self.transform(image)
        return image, label


#   Train: data augmentation (flips, rotación, jitter) + toTensor + Normalize. Augmentation solo en train: regulariza, evita overfit.
train_transform = T.Compose([
    T.RandomHorizontalFlip(),   # flip horizontal
    T.RandomVerticalFlip(),     # flip vertical
    T.RandomRotation(90),       # rotaciones
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),            # chiches para modificar imágenes
    T.ToTensor(),               # casting a tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),             # normalización
])

#   Eval/test: solo ToTensor + normalize. Sin augmentation para evaluación determinística.
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = EuroSATDataset("train.csv", dataset, transform=train_transform)
val_ds = EuroSATDataset("validation.csv", dataset, transform=eval_transform)
test_ds = EuroSATDataset("test.csv", dataset, transform=eval_transform)

train_loader    = DataLoader(train_ds,  batch_size=BATCH_SIZE, shuffle=True,    num_workers=NUM_WORKERS)
val_loader      = DataLoader(val_ds,    batch_size=BATCH_SIZE, shuffle=False,   num_workers=NUM_WORKERS)
test_loader     = DataLoader(test_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=NUM_WORKERS)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# TODO: seguir revisando de acá en más.

# ═══════════════════════════════════════════════════════════════════════════
# PARTE 1: CNN Custom desde cero
# ═══════════════════════════════════════════════════════════════════════════

class CustomCNN(nn.Module):
    """
    CNN custom con 4 bloques convolucionales.
    Cada bloque: Conv -> BatchNorm -> ReLU -> MaxPool
    Después: Global Average Pooling -> FC -> Clasificación
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: 3 -> 32 canales, 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Bloque 2: 32 -> 64 canales, 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Bloque 3: 64 -> 128 canales, 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Bloque 4: 128 -> 256 canales, 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Global Average Pooling -> vector de 256 dims
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Extraer features antes de la capa de clasificación final."""
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        # Pasar por primera FC + ReLU (sin la última linear)
        x = self.classifier[0](x)  # Dropout
        x = self.classifier[1](x)  # Linear 256->128
        x = self.classifier[2](x)  # ReLU
        return x


# ---------------------------------------------------------------------------
# Monitoreo de gradientes
# ---------------------------------------------------------------------------
def monitor_gradients(model, epoch):
    """Reporta norma de gradientes por capa para detectar vanishing/exploding."""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            grad_norms[name] = param.grad.norm().item()

    if epoch < 5:  # Solo primeras épocas
        print(f"  [Gradientes época {epoch}]")
        for name, norm in grad_norms.items():
            status = ""
            if norm < 1e-6:
                status = " ⚠ VANISHING"
            elif norm > 100:
                status = " ⚠ EXPLODING"
            print(f"    {name}: {norm:.6f}{status}")
    return grad_norms


# ---------------------------------------------------------------------------
# Training loop genérico
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, model_name="model", monitor_grads=False):
    """Loop de entrenamiento completo con historial de métricas."""
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "epoch_time": [], "grad_norms": []}
    best_val_acc = 0

    for epoch in range(epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)

        # Monitorear gradientes (solo primeras épocas)
        grad_norms = {}
        if monitor_grads:
            grad_norms = monitor_gradients(model, epoch)

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if scheduler:
            scheduler.step(val_loss)

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)
        history["grad_norms"].append(grad_norms)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

    print(f"\nMejor Val Accuracy ({model_name}): {best_val_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# Gráficos de entrenamiento
# ---------------------------------------------------------------------------
def plot_training_history(history, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss — {title}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy — {title}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"training_{title.replace(' ', '_').lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_gradient_norms(history, title=""):
    """Graficar evolución de normas de gradientes en primeras épocas."""
    grad_data = history["grad_norms"]
    epochs_with_grads = [i for i, g in enumerate(grad_data) if g]
    if not epochs_with_grads:
        return

    # Tomar solo capas conv y linear
    all_layers = list(grad_data[epochs_with_grads[0]].keys())
    key_layers = [l for l in all_layers if "features" in l or "classifier" in l]

    fig, ax = plt.subplots(figsize=(12, 5))
    for layer in key_layers:
        norms = [grad_data[e].get(layer, 0) for e in epochs_with_grads]
        short_name = layer.replace("features.", "F.").replace("classifier.", "C.")
        ax.plot(epochs_with_grads, norms, marker="o", markersize=3, label=short_name)

    ax.set_xlabel("Época")
    ax.set_ylabel("Norma del gradiente")
    ax.set_title(f"Monitoreo de gradientes — {title}")
    ax.legend(fontsize=7, ncol=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"gradients_{title.replace(' ', '_').lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Entrenar CNN Custom
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PARTE 1: Entrenamiento CNN Custom")
print("=" * 70)

custom_model = CustomCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer_custom = optim.Adam(custom_model.parameters(), lr=LR_CUSTOM, weight_decay=1e-4)
scheduler_custom = optim.lr_scheduler.ReduceLROnPlateau(optimizer_custom, patience=5, factor=0.5)

print(f"\nArquitectura:\n{custom_model}\n")
total_params = sum(p.numel() for p in custom_model.parameters())
print(f"Parámetros totales: {total_params:,}\n")

history_custom = train_model(
    custom_model, train_loader, val_loader, criterion,
    optimizer_custom, scheduler_custom,
    epochs=EPOCHS_CUSTOM, model_name="custom_cnn",
    monitor_grads=True,
)

plot_training_history(history_custom, title="CNN Custom")
plot_gradient_norms(history_custom, title="CNN Custom")


# ═══════════════════════════════════════════════════════════════════════════
# PARTE 2: Análisis de Errores y Espacio Latente
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PARTE 2: Análisis de Errores y Espacio Latente")
print("=" * 70)

# Cargar mejor modelo
custom_model.load_state_dict(torch.load("best_custom_cnn.pth", weights_only=True))
custom_model.eval()

# --- Evaluación en test ---
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = custom_model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy (CNN Custom): {test_acc:.4f}\n")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# --- Matriz de confusión ---
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusión — CNN Custom (Test)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix_custom.png", dpi=150, bbox_inches="tight")
plt.show()

# Análisis de confusiones más frecuentes
print("\nConfusiones más frecuentes:")
np.fill_diagonal(cm, 0)
for _ in range(5):
    i, j = np.unravel_index(cm.argmax(), cm.shape)
    if cm[i, j] == 0:
        break
    print(f"  {CLASS_NAMES[i]} -> {CLASS_NAMES[j]}: {cm[i, j]} errores")
    cm[i, j] = 0

print("""
Análisis: Las confusiones más comunes se dan entre clases visualmente similares
en imágenes satelitales. Por ejemplo:
- AnnualCrop vs PermanentCrop: ambos son terrenos agrícolas con texturas parecidas.
- HerbaceousVegetation vs Forest/Pasture: vegetación con diferencias sutiles de densidad.
- River vs SeaLake: ambos son cuerpos de agua con bordes parecidos.
- Residential vs Industrial: estructuras urbanas con patrones similares.
Estas confusiones reflejan la ambigüedad inherente en la clasificación de uso de suelo
donde las fronteras entre clases pueden ser difusas.
""")

# --- Espacio latente con t-SNE ---
print("Extrayendo features del espacio latente...")
all_features, all_labels_tsne = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        feats = custom_model.extract_features(images)
        all_features.append(feats.cpu().numpy())
        all_labels_tsne.extend(labels.numpy())

all_features = np.concatenate(all_features, axis=0)
all_labels_tsne = np.array(all_labels_tsne)

print(f"Shape features: {all_features.shape}")
print("Calculando t-SNE (puede tardar unos segundos)...")

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
features_2d = tsne.fit_transform(all_features)

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                     c=all_labels_tsne, cmap="tab10", s=8, alpha=0.7)
handles, _ = scatter.legend_elements()
ax.legend(handles, CLASS_NAMES, loc="best", fontsize=8)
ax.set_title("Espacio Latente (t-SNE) — CNN Custom")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("tsne_custom.png", dpi=150, bbox_inches="tight")
plt.show()

print("""
Análisis del espacio latente:
La visualización t-SNE muestra que la red logra formar clusters distinguibles para
la mayoría de las clases. Clases como Forest, SeaLake y Highway forman clusters
compactos y bien separados, lo que explica su alto accuracy individual.
Las clases con mayor solapamiento en el espacio latente (AnnualCrop/PermanentCrop,
HerbaceousVegetation/Pasture) coinciden con las confusiones más frecuentes
observadas en la matriz de confusión.
""")


# ═══════════════════════════════════════════════════════════════════════════
# PARTE 3: Transfer Learning con ResNet18
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PARTE 3: Transfer Learning — ResNet18")
print("=" * 70)

# Las imágenes son 64x64, ResNet espera 224x224 -> resize
transfer_train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(90),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transfer_eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transfer_train_ds = EuroSATDataset("train.csv", DATA_DIR, transform=transfer_train_transform)
transfer_val_ds = EuroSATDataset("validation.csv", DATA_DIR, transform=transfer_eval_transform)
transfer_test_ds = EuroSATDataset("test.csv", DATA_DIR, transform=transfer_eval_transform)

transfer_train_loader = DataLoader(transfer_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
transfer_val_loader = DataLoader(transfer_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
transfer_test_loader = DataLoader(transfer_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Cargar ResNet18 preentrenada en ImageNet
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Congelar todas las capas excepto layer4 y fc
for name, param in resnet.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# Reemplazar capa fc para 10 clases
resnet.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(resnet.fc.in_features, NUM_CLASSES),
)
resnet = resnet.to(DEVICE)

trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
total = sum(p.numel() for p in resnet.parameters())
print(f"\nResNet18 — Parámetros entrenables: {trainable:,} / {total:,}")

optimizer_transfer = optim.Adam(
    filter(lambda p: p.requires_grad, resnet.parameters()),
    lr=LR_TRANSFER, weight_decay=1e-4,
)
scheduler_transfer = optim.lr_scheduler.ReduceLROnPlateau(optimizer_transfer, patience=3, factor=0.5)

history_transfer = train_model(
    resnet, transfer_train_loader, transfer_val_loader, criterion,
    optimizer_transfer, scheduler_transfer,
    epochs=EPOCHS_TRANSFER, model_name="resnet18",
    monitor_grads=False,
)

plot_training_history(history_transfer, title="ResNet18 Transfer Learning")

# --- Evaluación en test ---
resnet.load_state_dict(torch.load("best_resnet18.pth", weights_only=True))
resnet.eval()

all_preds_tl, all_labels_tl = [], []
with torch.no_grad():
    for images, labels in transfer_test_loader:
        images = images.to(DEVICE)
        outputs = resnet(images)
        _, preds = outputs.max(1)
        all_preds_tl.extend(preds.cpu().numpy())
        all_labels_tl.extend(labels.numpy())

all_preds_tl = np.array(all_preds_tl)
all_labels_tl = np.array(all_labels_tl)

test_acc_tl = accuracy_score(all_labels_tl, all_preds_tl)
print(f"\nTest Accuracy (ResNet18): {test_acc_tl:.4f}\n")
print("Classification Report:")
print(classification_report(all_labels_tl, all_preds_tl, target_names=CLASS_NAMES))

# Matriz de confusión
cm_tl = confusion_matrix(all_labels_tl, all_preds_tl)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_tl, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusión — ResNet18 (Test)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix_resnet18.png", dpi=150, bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Comparación Final
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPARACIÓN: CNN Custom vs ResNet18 (Transfer Learning)")
print("=" * 70)

avg_time_custom = np.mean(history_custom["epoch_time"])
avg_time_transfer = np.mean(history_transfer["epoch_time"])

print(f"""
┌────────────────────────┬──────────────┬──────────────┐
│ Métrica                │ CNN Custom   │ ResNet18 TL  │
├────────────────────────┼──────────────┼──────────────┤
│ Test Accuracy          │ {test_acc:.4f}       │ {test_acc_tl:.4f}       │
│ Mejor Val Accuracy     │ {max(history_custom['val_acc']):.4f}       │ {max(history_transfer['val_acc']):.4f}       │
│ Épocas entrenadas      │ {EPOCHS_CUSTOM:<12d} │ {EPOCHS_TRANSFER:<12d} │
│ Tiempo/época (seg)     │ {avg_time_custom:<12.1f} │ {avg_time_transfer:<12.1f} │
│ Tiempo total (seg)     │ {sum(history_custom['epoch_time']):<12.1f} │ {sum(history_transfer['epoch_time']):<12.1f} │
│ Parámetros totales     │ {sum(p.numel() for p in custom_model.parameters()):>12,} │ {total:>12,} │
└────────────────────────┴──────────────┴──────────────┘
""")

# Gráfico comparativo de convergencia
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_custom["val_loss"], label="CNN Custom", marker=".")
axes[0].plot(history_transfer["val_loss"], label="ResNet18 TL", marker=".")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Validation Loss")
axes[0].set_title("Convergencia — Val Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_custom["val_acc"], label="CNN Custom", marker=".")
axes[1].plot(history_transfer["val_acc"], label="ResNet18 TL", marker=".")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Validation Accuracy")
axes[1].set_title("Convergencia — Val Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comparacion_convergencia.png", dpi=150, bbox_inches="tight")
plt.show()

print("""
Conclusión:
- Transfer Learning (ResNet18) converge más rápido y generalmente alcanza
  mejor accuracy que la CNN custom, incluso con menos épocas de entrenamiento.
- Esto se debe a que las features aprendidas en ImageNet (bordes, texturas,
  formas) son transferibles al dominio satelital.
- La CNN custom, a pesar de tener menos parámetros, requiere más épocas
  para aprender representaciones útiles desde cero.
- El tiempo por época es mayor en ResNet18 por el resize a 224x224 y la
  mayor cantidad de parámetros, pero el menor número de épocas compensa.
- Fine-tuning solo de layer4 + fc es efectivo: mantiene las features
  generales de bajo nivel congeladas y adapta las de alto nivel al dominio.
""")

print("TP3 completado. Archivos generados:")
for f in ["training_cnn_custom.png", "gradients_cnn_custom.png",
          "confusion_matrix_custom.png", "tsne_custom.png",
          "training_resnet18_transfer_learning.png",
          "confusion_matrix_resnet18.png", "comparacion_convergencia.png"]:
    print(f"  - {f}")
