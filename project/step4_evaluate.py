import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Налаштування
# =========================

SPEC_DIR = "data/spectrograms"
BATCH_SIZE = 32
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# =========================
# Трансформації
# =========================

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# =========================
# Dataset і DataLoader
# =========================

full_ds = datasets.ImageFolder(
    SPEC_DIR,
    transform=val_tf
)

CLASS_NAMES = full_ds.classes
NUM_CLASSES = len(CLASS_NAMES)

n = len(full_ds)

n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    full_ds,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Test samples: {n_test}")

# =========================
# Створення моделі
# =========================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

# Завантаження ваг
model.load_state_dict(
    torch.load("best_model.pth", map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

print("Модель успішно завантажена.")

# =========================
# Тестування
# =========================

all_preds = []
all_labels = []

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(DEVICE)

        outputs = model(imgs)

        preds = outputs.argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# =========================
# Classification Report
# =========================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES
    )
)

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("Матриця помилок")
plt.xlabel("Передбачений клас")
plt.ylabel("Справжній клас")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=150
)

plt.show()

# =========================
# Accuracy
# =========================

test_acc = np.mean(
    np.array(all_preds) == np.array(all_labels)
)

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")