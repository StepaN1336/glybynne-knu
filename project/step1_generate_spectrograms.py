import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

AUDIO_DIR = "data/Data/genres_original"
SPEC_DIR  = "data/spectrograms"
SR        = 22050
DURATION  = 30   # секунд
HOP       = 512

def save_spectrogram(audio_path, out_path):
    y, sr = librosa.load(audio_path, sr=SR, duration=DURATION, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=HOP)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sr, hop_length=HOP, ax=ax)
    plt.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

for genre in os.listdir(AUDIO_DIR):
    genre_path = Path(AUDIO_DIR) / genre
    if not genre_path.is_dir():
        continue
    out_genre = Path(SPEC_DIR) / genre
    out_genre.mkdir(parents=True, exist_ok=True)
    for audio_file in genre_path.glob("*.wav"):
        out_file = out_genre / (audio_file.stem + ".png")
        if out_file.exists():
            continue
        try:
            save_spectrogram(str(audio_file), str(out_file))
        except Exception as e:
            print(f"Skip {audio_file.name}: {e}")

print("Спектрограми згенеровано!")