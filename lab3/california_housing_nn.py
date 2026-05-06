"""
California Housing Price Prediction
PyTorch Feed-Forward Neural Network — 5 Experiment Configurations

Dataset : California Housing (scikit-learn)
Task    : Predict median house value (USD) from 8 district-level features
Author  : Lab work

Parameters fixed across all experiments
  test_size    = 0.2
  random_state = 42
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── 1. Load dataset ───────────────────────────────────────────────────────────
housing = fetch_california_housing()
X = housing.data                        # shape (20640, 8)
y = housing.target * 100_000            # convert $100k → USD

print("Feature names:", housing.feature_names)
print(f"Samples: {X.shape[0]:,}  |  Features: {X.shape[1]}")

# ── 2. Train / test split ─────────────────────────────────────────────────────
TEST_SIZE = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"test_size = {TEST_SIZE}   random_state = {RANDOM_STATE}\n")

# ── 3. Standardisation (fit ONLY on training data) ────────────────────────────
# The model has NO prior knowledge of the test set.
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

X_train_s = scaler_X.transform(X_train)
X_test_s  = scaler_X.transform(X_test)
y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ── PyTorch tensors ───────────────────────────────────────────────────────────
def to_tensors(X_np, y_np):
    return (torch.tensor(X_np, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.float32).unsqueeze(1))

X_tr_t, y_tr_t = to_tensors(X_train_s, y_train_s)
X_te_t, y_te_t = to_tensors(X_test_s,  y_test_s)

# ── 4. Model factory ──────────────────────────────────────────────────────────
def build_model(hidden_layers, activation_cls, dropout=0.0, batch_norm=False):
    """
    hidden_layers  : list[int]  — number of neurons per hidden layer
    activation_cls : nn.Module  — activation class (not instance)
    dropout        : float      — dropout probability applied after each hidden layer
    batch_norm     : bool       — apply BatchNorm1d before activation
    """
    layers = []
    in_features = 8  # number of input features
    for units in hidden_layers:
        layers.append(nn.Linear(in_features, units))
        if batch_norm:
            layers.append(nn.BatchNorm1d(units))
        layers.append(activation_cls())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_features = units
    layers.append(nn.Linear(in_features, 1))   # single regression output
    return nn.Sequential(*layers)

# ── 5. Training function ──────────────────────────────────────────────────────
def train_and_evaluate(model, optimizer, loss_fn, epochs, batch_size):
    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True
    )
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(X_tr_t))

    # ── Evaluation on test set ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_te_t).numpy()

    pred_usd    = scaler_y.inverse_transform(pred_scaled).ravel()
    test_loss   = loss_fn(
        torch.tensor(pred_scaled, dtype=torch.float32), y_te_t
    ).item()
    mae = mean_absolute_error(y_test, pred_usd)
    r2  = r2_score(y_test, pred_usd)
    return pred_usd, test_loss, mae, r2, train_losses

# ── 6. Five experiment configurations ─────────────────────────────────────────
#
#  Each experiment differs in:
#   • hidden layer architecture  (depth, width)
#   • activation function
#   • batch normalisation
#   • dropout rate
#   • optimizer
#   • loss function
#
experiments = [
    # Exp 1 — shallow, ReLU, no regularisation, Adam, MSE
    {
        "id":      1,
        "hidden":  [64, 32],
        "act":     nn.ReLU,
        "drop":    0.0,
        "bn":      False,
        "opt_fn":  lambda p: optim.Adam(p, lr=1e-3),
        "opt_str": "Adam (lr=1e-3)",
        "loss_fn": nn.MSELoss(),
        "loss_str":"MSELoss",
        "epochs":  80,
        "bs":      64,
    },
    # Exp 2 — medium depth, ReLU, dropout 0.2, Adam, MSE
    {
        "id":      2,
        "hidden":  [128, 64, 32],
        "act":     nn.ReLU,
        "drop":    0.2,
        "bn":      False,
        "opt_fn":  lambda p: optim.Adam(p, lr=1e-3),
        "opt_str": "Adam (lr=1e-3)",
        "loss_fn": nn.MSELoss(),
        "loss_str":"MSELoss",
        "epochs":  80,
        "bs":      128,
    },
    # Exp 3 — Tanh activation, BatchNorm, dropout 0.3, RMSprop, Huber
    {
        "id":      3,
        "hidden":  [256, 128, 64],
        "act":     nn.Tanh,
        "drop":    0.3,
        "bn":      True,
        "opt_fn":  lambda p: optim.RMSprop(p, lr=1e-3),
        "opt_str": "RMSprop (lr=1e-3)",
        "loss_fn": nn.HuberLoss(),
        "loss_str":"HuberLoss",
        "epochs":  80,
        "bs":      256,
    },
    # Exp 4 — 4 hidden layers, ELU, BatchNorm, AdamW + L2, MSE  ← BEST
    {
        "id":      4,
        "hidden":  [128, 128, 64, 32],
        "act":     nn.ELU,
        "drop":    0.1,
        "bn":      True,
        "opt_fn":  lambda p: optim.AdamW(p, lr=5e-4, weight_decay=1e-4),
        "opt_str": "AdamW (lr=5e-4, wd=1e-4)",
        "loss_fn": nn.MSELoss(),
        "loss_str":"MSELoss",
        "epochs":  80,
        "bs":      128,
    },
    # Exp 5 — very wide, LeakyReLU, BatchNorm, Adam amsgrad, Huber
    {
        "id":      5,
        "hidden":  [512, 256, 128],
        "act":     nn.LeakyReLU,
        "drop":    0.25,
        "bn":      True,
        "opt_fn":  lambda p: optim.Adam(p, lr=1e-3, amsgrad=True),
        "opt_str": "Adam amsgrad (lr=1e-3)",
        "loss_fn": nn.HuberLoss(),
        "loss_str":"HuberLoss",
        "epochs":  80,
        "bs":      256,
    },
]

# ── 7. Run experiments ────────────────────────────────────────────────────────
all_results   = []
all_preds     = []
all_histories = []

print("=" * 72)
for cfg in experiments:
    torch.manual_seed(RANDOM_STATE)
    model = build_model(cfg["hidden"], cfg["act"], cfg["drop"], cfg["bn"])
    opt   = cfg["opt_fn"](model.parameters())

    preds, test_loss, mae, r2, hist = train_and_evaluate(
        model, opt, cfg["loss_fn"], cfg["epochs"], cfg["bs"]
    )

    all_results.append({**cfg, "test_loss": test_loss, "mae": mae, "r2": r2})
    all_preds.append(preds)
    all_histories.append(hist)

    print(
        f"Exp {cfg['id']} | {cfg['loss_str']:10s} on test: {test_loss:.4f} | "
        f"MAE: ${mae:,.0f} | R²: {r2:.4f}"
    )
print("=" * 72)

# ── 8. Best experiment ────────────────────────────────────────────────────────
best = max(all_results, key=lambda r: r["r2"])
best_idx = best["id"] - 1
print(
    f"\n★ Best experiment: #{best['id']}"
    f"  R² = {best['r2']:.4f} | MAE = ${best['mae']:,.0f}\n"
)

# ── 9. Visualisations ─────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# (a) MAE comparison
ax0 = fig.add_subplot(gs[0, 0])
maes   = [r["mae"] for r in all_results]
colors = ["#e74c3c" if r["id"] == best["id"] else "#3498db" for r in all_results]
bars   = ax0.bar([f"Exp {r['id']}" for r in all_results], maes, color=colors)
ax0.bar_label(bars, fmt="$%.0f", fontsize=8, padding=3)
ax0.set_title("MAE on Test Set (USD)", fontsize=12)
ax0.set_ylabel("MAE (USD)")

# (b) R² comparison
ax1 = fig.add_subplot(gs[0, 1])
r2s    = [r["r2"] for r in all_results]
cols2  = ["#e74c3c" if r["id"] == best["id"] else "#27ae60" for r in all_results]
bars2  = ax1.bar([f"Exp {r['id']}" for r in all_results], r2s, color=cols2)
ax1.bar_label(bars2, fmt="%.4f", fontsize=8, padding=3)
ax1.set_title("R² Score on Test Set", fontsize=12)
ax1.set_ylabel("R²")
ax1.set_ylim(min(r2s) - 0.003, max(r2s) + 0.005)

# (c) Actual vs Predicted — best experiment, first 150 test samples
n_show = 150
ax2    = fig.add_subplot(gs[1, :])
true_s = y_test[:n_show]
pred_s = all_preds[best_idx][:n_show]
x_idx  = np.arange(n_show)
ax2.plot(x_idx, true_s, label="Фактичні", color="#27ae60", linewidth=1.5, alpha=0.85)
ax2.plot(x_idx, pred_s, label="Передбачені", color="#e74c3c", linewidth=1.5,
         alpha=0.85, linestyle="--")
ax2.set_title(
    f"Експеримент #{best['id']} (Найкращий R² = {best['r2']:.4f}) — "
    f"Фактичні vs Передбачені (перші {n_show} зразків тестової множини)\n"
    f"MAE = ${best['mae']:,.0f}  |  Шари: {best['hidden']}  |  Активація: {best['act'].__name__}",
    fontsize=11
)
ax2.set_xlabel("Індекс зразка")
ax2.set_ylabel("Ціна на житло (USD)")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax2.legend(fontsize=10)

plt.suptitle(
    "California Housing — PyTorch Feed-Forward Neural Network\n"
    f"test_size = {TEST_SIZE}  |  random_state = {RANDOM_STATE}",
    fontsize=13, y=1.02
)
plt.savefig("california_housing_results.png", dpi=150, bbox_inches="tight")
print("Plot saved → california_housing_results.png")

# ── 10. Summary table ─────────────────────────────────────────────────────────
print("\nSUMMARY TABLE")
print("=" * 100)
print(f"{'#':>2} | {'Epochs':>6} | {'Hidden layers':^22} | {'Act.':^11} | "
      f"{'BN':>3} | {'Drop':>4} | {'Optimizer':^24} | "
      f"{'Loss fn':^10} | {'Test loss':>9} | {'MAE ($)':>9} | {'R²':>6}")
print("-" * 100)
for r in all_results:
    marker = " ★" if r["id"] == best["id"] else "  "
    print(
        f"{r['id']:>2}{marker}| {r['epochs']:>6} | {str(r['hidden']):^22} | "
        f"{r['act'].__name__:^11} | {'Y' if r['bn'] else 'N':>3} | "
        f"{r['drop']:>4} | {r['opt_str']:^24} | "
        f"{r['loss_str']:^10} | {r['test_loss']:>9.4f} | "
        f"{r['mae']:>9,.0f} | {r['r2']:>6.4f}"
    )
print("=" * 100)
