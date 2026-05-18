import os
import re
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, GRU, Dense, Dropout, Bidirectional,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D,
    Input, Concatenate, LayerNormalization, MultiHeadAttention,
    SpatialDropout1D, BatchNormalization
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('punkt_tab')

# ─── 1. Завантаження даних ────────────────────────────────────────────────────

def load_reviews(path):
    texts, labels = [], []
    for label, folder in [(1, 'pos'), (0, 'neg')]:
        folder_path = os.path.join(path, folder)
        for fname in os.listdir(folder_path):
            if fname.endswith('.txt'):
                with open(os.path.join(folder_path, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
    return texts, labels

print("Завантаження датасету...")
train_texts, train_labels = load_reviews('aclImdb/train')
test_texts,  test_labels  = load_reviews('aclImdb/test')
print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

# ─── 2. Токенізація та побудова словника ──────────────────────────────────────

def clean(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower()

print("Токенізація...")
train_tokens = [word_tokenize(clean(t)) for t in train_texts]
test_tokens  = [word_tokenize(clean(t)) for t in test_texts]

VOCAB_SIZE = 20000
MAX_LEN    = 300

counter  = Counter(tok for seq in train_tokens for tok in seq)
vocab    = ['<PAD>', '<UNK>'] + [w for w, _ in counter.most_common(VOCAB_SIZE - 2)]
word2idx = {w: i for i, w in enumerate(vocab)}

with open('tokenizer_v2.pkl', 'wb') as f:
    pickle.dump(word2idx, f)
print("Словник збережено у tokenizer_v2.pkl")

def encode_and_pad(token_lists, w2i, max_len):
    encoded = [[w2i.get(t, 1) for t in seq] for seq in token_lists]
    return tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=max_len, padding='post', truncating='post')

X_train = encode_and_pad(train_tokens, word2idx, MAX_LEN)
X_test  = encode_and_pad(test_tokens,  word2idx, MAX_LEN)
y_train = np.array(train_labels)
y_test  = np.array(test_labels)

# ─── 3. Нові конфігурації ────────────────────────────────────────────────────

def build_model_1():
    """
    Config 1: BiGRU + GlobalMaxPooling
    GRU навчається швидше за LSTM, менше параметрів → менше overfit.
    GlobalMaxPooling витягує найсильніший сигнал по всій послідовності.
    """
    m = Sequential([
        Embedding(VOCAB_SIZE, 128, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Bidirectional(GRU(64, return_sequences=True, dropout=0.2)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_model_2():
    """
    Config 2: Conv1D → BiLSTM + GlobalMaxPooling
    Conv1D спершу вивчає локальні n-gram ознаки (як bag-of-n-grams),
    потім BiLSTM моделює глобальний контекст поверх них.
    """
    m = Sequential([
        Embedding(VOCAB_SIZE, 128, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_model_3():
    """
    Config 3: BiLSTM(128) + GlobalMaxPooling (простий, але потужний baseline)
    Більший BiLSTM (128 одиниць на напрямок), довший контекст (MAX_LEN=300).
    """
    m = Sequential([
        Embedding(VOCAB_SIZE, 128, input_length=MAX_LEN),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_model_4():
    """
    Config 4: Multi-scale Conv1D (3 розміри ядра) → BiLSTM
    Паралельні conv-фільтри 2/3/5 захоплюють bi-grams, tri-grams і 5-grams.
    Функціональне API (не Sequential).
    """
    inp = Input(shape=(MAX_LEN,))
    emb = Embedding(VOCAB_SIZE, 128)(inp)
    emb = SpatialDropout1D(0.2)(emb)

    # три гілки з різними ядрами
    branches = []
    for kernel in [2, 3, 5]:
        x = Conv1D(64, kernel_size=kernel, activation='relu', padding='same')(emb)
        branches.append(x)

    merged = Concatenate()(branches)
    x = Bidirectional(LSTM(64, dropout=0.2, return_sequences=True))(merged)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model_5():
    """
    Config 5: BiLSTM(128) → BiLSTM(64) + MaxPool + AvgPool (об'єднані)
    Конкатенація GlobalMax і GlobalAvg дає ширший фінальний вектор:
    MaxPool = «що є», AvgPool = «наскільки часто».
    """
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(VOCAB_SIZE, 256)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(64,  return_sequences=True, dropout=0.2))(x)

    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = Concatenate()([max_pool, avg_pool])

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(5e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

configs = [
    (build_model_1, 'Adam 1e-3', 'Config 1: BiGRU+MaxPool'),
    (build_model_2, 'Adam 1e-3', 'Config 2: Conv1D→BiLSTM+MaxPool'),
    (build_model_3, 'Adam 1e-3', 'Config 3: BiLSTM(128)+MaxPool'),
    (build_model_4, 'Adam 1e-3', 'Config 4: MultiScaleConv→BiLSTM'),
    (build_model_5, 'Adam 5e-4', 'Config 5: BiLSTM²+MaxPool+AvgPool'),
]

# ─── 4. Навчання ─────────────────────────────────────────────────────────────

reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                               patience=1, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3,
                           restore_best_weights=True, verbose=1)

results   = []
histories = []

for i, (builder, opt_name, desc) in enumerate(configs, 1):
    print(f"\n{'='*60}")
    print(f"Конфігурація {i}: {desc}")
    print('='*60)
    model = builder()
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

    results.append({
        'config': i, 'desc': desc, 'optimizer': opt_name,
        'epochs': len(history.history['loss']),
        'test_loss': loss, 'test_acc': acc, 'history': history
    })
    histories.append(history)

    if acc == max(r['test_acc'] for r in results):
        model.save('best_model_v2.keras')
        print("  → Збережено як best_model_v2.keras")

# ─── 5. Таблиця результатів ───────────────────────────────────────────────────

print("\n" + "="*95)
print(f"{'№':>3} | {'Опис':<35} | {'Optimizer':<12} | {'Epochs':>6} | {'Loss':>8} | {'Accuracy':>8}")
print("-"*95)
for r in results:
    marker = " ★" if r['test_acc'] == max(x['test_acc'] for x in results) else ""
    print(f"{r['config']:>3} | {r['desc']:<35} | {r['optimizer']:<12} | "
          f"{r['epochs']:>6} | {r['test_loss']:>8.4f} | {r['test_acc']:>8.4f}{marker}")
print("="*95)

best = max(results, key=lambda r: r['test_acc'])
print(f"\nНайкраща конфігурація: №{best['config']} ({best['desc']})")
print(f"  Accuracy: {best['test_acc']:.4f}, Loss: {best['test_loss']:.4f}")

# ─── 6. Графіки ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (r, h) in enumerate(zip(results, histories)):
    ax = axes[i]
    ax.plot(h.history['accuracy'],     label='Train Acc', linewidth=2)
    ax.plot(h.history['val_accuracy'], label='Val Acc',   linewidth=2, linestyle='--')
    ax.set_title(f"{r['desc']}\nTest Acc: {r['test_acc']:.4f}", fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig('training_curves_v2.png', dpi=120)
print("\nГрафіки збережено у training_curves_v2.png")

# ── 7. Графік найкращої моделі ─────────────────────────────────────────────────

best_history = best['history']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Найкраща модель: {best['desc']}", fontsize=13)

# Функція втрат
ax1.plot(best_history.history['loss'],     label='Train Loss',      color='steelblue')
ax1.plot(best_history.history['val_loss'], label='Validation Loss',  color='tomato')
ax1.set_title('Функція втрат (Binary Cross-Entropy)')
ax1.set_xlabel('Епоха')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Точність
ax2.plot(best_history.history['accuracy'],     label='Train Accuracy',     color='steelblue')
ax2.plot(best_history.history['val_accuracy'], label='Validation Accuracy', color='tomato')
ax2.set_title('Точність (Accuracy)')
ax2.set_xlabel('Епоха')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("Графік збережено: training_curves.png")