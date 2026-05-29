import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

SPEC_DIR   = "data/spectrograms"
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-3
IMG_SIZE   = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Device: {DEVICE}")

    # Трансформації
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_ds = datasets.ImageFolder(SPEC_DIR, transform=train_tf)

    n = len(full_ds)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    val_ds.dataset.transform = val_tf
    test_ds.dataset.transform = val_tf

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    CLASS_NAMES = full_ds.classes
    NUM_CLASSES = len(CLASS_NAMES)

    print(f"Класів: {NUM_CLASSES}")

    # Модель
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.set_grad_enabled(train):
            for imgs, labels in loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                out = model(imgs)
                loss = criterion(out, labels)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                correct += (out.argmax(1) == labels).sum().item()
                total += imgs.size(0)

        return total_loss / total, correct / total

    best_val_acc = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        vl_loss, vl_acc = run_epoch(val_loader, train=False)

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}"
        )

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Функція втрат")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_title("Точність")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()