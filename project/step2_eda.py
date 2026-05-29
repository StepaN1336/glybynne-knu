import os
from collections import Counter
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from pathlib import Path

AUDIO_DIR = "data/Data/genres_original"
SPEC_DIR  = "data/spectrograms"

# Розподіл класів
genres = os.listdir(SPEC_DIR)
counts = {g: len(list(Path(SPEC_DIR, g).glob("*.png"))) for g in genres}

plt.figure(figsize=(10, 4))
plt.bar(counts.keys(), counts.values(), color="steelblue")
plt.title("Розподіл класів у датасеті GTZAN")
plt.xlabel("Жанр")
plt.ylabel("Кількість треків")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()
print(counts)

# Приклади спектрограм по одній з кожного жанру
fig, axes = plt.subplots(2, 5, figsize=(16, 6))
for ax, genre in zip(axes.flat, sorted(genres)):
    img_path = next(Path(SPEC_DIR, genre).glob("*.png"))
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_title(genre, fontsize=10)
    ax.axis("off")
plt.suptitle("Приклади мел-спектрограм по жанрах")
plt.tight_layout()
plt.savefig("spectrogram_examples.png", dpi=150)
plt.show()

# Waveform + спектрограма для одного треку
sample_wav = str(next(Path(AUDIO_DIR, "jazz").glob("*.wav")))
y, sr = librosa.load(sample_wav, sr=22050, duration=30)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
librosa.display.waveshow(y, sr=sr, ax=axes[0])
axes[0].set_title("Waveform — Jazz")

mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)
librosa.display.specshow(mel_db, sr=sr, ax=axes[1], x_axis="time", y_axis="mel")
axes[1].set_title("Мел-спектрограма — Jazz")
plt.tight_layout()
plt.savefig("waveform_vs_spectrogram.png", dpi=150)
plt.show()