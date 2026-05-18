import pickle
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf

nltk.download('punkt')
nltk.download('punkt_tab')

# Завантажуємо токенізатор і модель
with open('tokenizer_v2.pkl', 'rb') as f:
    word2idx = pickle.load(f)

model = tf.keras.models.load_model('best_model_v2.keras')
MAX_LEN = 200

def predict(texts):
    def clean(t):
        t = re.sub(r'<.*?>', ' ', t)
        t = re.sub(r'[^a-zA-Z\s]', ' ', t)
        return t.lower()

    tokenized = [word_tokenize(clean(t)) for t in texts]
    encoded   = [[word2idx.get(tok, 1) for tok in seq] for seq in tokenized]
    padded    = tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model.predict(padded, verbose=0).flatten()
    return probs

# ── П. 1.2.4: Власні рецензії ──────────────────────────────────────────────────
my_reviews = [
    "This film was absolutely wonderful. The acting and story were both superb.",
    "Terrible movie. Waste of time, awful script and poor direction.",
    "An average film, nothing special but not terrible either.",
    "One of the greatest films I have ever seen. A true masterpiece.",
]

print("=" * 60)
print("П. 1.2.4 — Власні рецензії")
print("=" * 60)
probs = predict(my_reviews)
for text, p in zip(my_reviews, probs):
    sentiment = "ПОЗИТИВНА ✅" if p >= 0.5 else "НЕГАТИВНА ❌"
    print(f"\nТекст : {text}")
    print(f"Вихід : {p:.4f}  →  {sentiment}")

# ── П. 1.2.5: Задані висловлювання ────────────────────────────────────────────
test_sentences = [
    "I expected to hate it, but it was actually the best movie of the year.",
    "The plot was as deep as a puddle.",
    "It was not a bad film, but certainly not a great one either.",
]

print("\n" + "=" * 60)
print("П. 1.2.5 — Задані висловлювання")
print("=" * 60)
probs2 = predict(test_sentences)
for text, p in zip(test_sentences, probs2):
    sentiment = "ПОЗИТИВНА ✅" if p >= 0.5 else "НЕГАТИВНА ❌"
    print(f"\nТекст : {text}")
    print(f"Вихід : {p:.4f}  →  {sentiment}")

print("\n--- Аналіз ---")
print("Речення 1: модель має розпізнати іронічний зворот 'expected to hate → best movie'.")
print("Речення 2: саркастичний негатив ('as deep as a puddle').")
print("Речення 3: нейтральне/змішане висловлювання — цікаво, куди схилиться модель.")