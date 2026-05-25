import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Data ─────────────────────────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# ─── Models ───────────────────────────────────────────────────────────────────

# Experiment 1: 2 conv blocks, Adam
class CNN_Exp1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, 100),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# Experiment 2: 3 conv blocks, more filters, Adam
class CNN_Exp2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(),
            nn.Linear(1024, 100),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# Experiment 3: 3 blocks + Dropout, SGD with momentum
class CNN_Exp3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 100),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# Experiment 4: 4 blocks + BatchNorm, Adam
class CNN_Exp4(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(),
            nn.Linear(1024, 100),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# Experiment 5: 4 blocks + BatchNorm + AvgPool in last block, Adam
class CNN_Exp5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(),
            nn.Linear(1024, 100),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ─── Train / Eval ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
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
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total

# ─── Run experiment ───────────────────────────────────────────────────────────

def run_experiment(exp_num, model, optimizer, train_loader, test_loader):
    print(f"\n{'='*60}")
    print(f"  Experiment {exp_num}")
    print(f"{'='*60}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    best_val_loss = float('inf')
    no_improve    = 0
    best_epoch    = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = eval_epoch(model,  test_loader,  criterion)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        scheduler.step(vl_loss)

        print(f"Epoch {epoch:2d}/{MAX_EPOCHS} | "
              f"Train loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"Val loss: {vl_loss:.4f} acc: {vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            no_improve    = 0
            best_epoch    = epoch
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    final_val_loss = val_losses[-1]
    final_val_acc  = val_accs[-1]
    print(f"\nFinal -> Val Loss: {final_val_loss:.4f} | Val Acc: {final_val_acc:.4f}")

    return {
        'epochs':         len(train_losses),
        'train_losses':   train_losses,
        'val_losses':     val_losses,
        'train_accs':     train_accs,
        'val_accs':       val_accs,
        'final_val_loss': final_val_loss,
        'final_val_acc':  final_val_acc,
    }

# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_history(history, exp_num):
    epochs = range(1, history['epochs'] + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Experiment {exp_num}', fontsize=14)

    ax1.plot(epochs, history['train_losses'], label='Train')
    ax1.plot(epochs, history['val_losses'],   label='Validation')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history['train_accs'], label='Train')
    ax2.plot(epochs, history['val_accs'],   label='Validation')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'exp{exp_num}_history.png', dpi=150)
    plt.show()
    print(f"Saved: exp{exp_num}_history.png")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    train_dataset = datasets.CIFAR100(root='./data', train=True,  download=True, transform=transform_train)
    test_dataset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    experiments = [
        (1, CNN_Exp1(), lambda m: optim.Adam(m.parameters(), lr=0.001)),
        (2, CNN_Exp2(), lambda m: optim.Adam(m.parameters(), lr=0.001)),
        (3, CNN_Exp3(), lambda m: optim.SGD(m.parameters(),  lr=0.01, momentum=0.9, weight_decay=1e-4)),
        (4, CNN_Exp4(), lambda m: optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)),
        (5, CNN_Exp5(), lambda m: optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)),
    ]

    results = {}

    for exp_num, model, opt_fn in experiments:
        optimizer = opt_fn(model)
        history = run_experiment(exp_num, model, optimizer, train_loader, test_loader)
        results[exp_num] = history

    # Summary table
    print("\n" + "="*60)
    print("  SUMMARY TABLE")
    print("="*60)
    print(f"{'Exp':>4} | {'Epochs':>6} | {'Val Loss':>9} | {'Val Acc':>8}")
    print("-"*40)
    for exp_num, h in results.items():
        print(f"{exp_num:>4} | {h['epochs']:>6} | {h['final_val_loss']:>9.4f} | {h['final_val_acc']:>8.4f}")

    # Plot best experiment
    best_exp = max(results, key=lambda k: results[k]['final_val_acc'])
    print(f"\nBest experiment: {best_exp} (acc={results[best_exp]['final_val_acc']:.4f})")
    plot_history(results[best_exp], best_exp)