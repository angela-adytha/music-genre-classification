import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm

RAW_DATASET_BASE = 'dataset/raw'
WAVEFORM_BASE = 'waveforms'
SAMPLE_RATE = 22050

SUBSETS = [
    'sekunder',
    'primer/0:30-1:00',
    'primer/1:00-1:30',
    'primer/1:30-2:00',
    'primer/2:00-2:30',
    'primer/2:30-3:00',
    'primer/reff',
]

for subset in SUBSETS:
    RAW_DATASET_PATH = os.path.join(RAW_DATASET_BASE, subset)
    WAVEFORM_PATH = os.path.join(WAVEFORM_BASE, subset)

    os.makedirs(WAVEFORM_PATH, exist_ok=True)

    genres = sorted([
        folder for folder in os.listdir(RAW_DATASET_PATH)
        if os.path.isdir(os.path.join(RAW_DATASET_PATH, folder))
    ])

    for genre in genres:
        genre_path = os.path.join(RAW_DATASET_PATH, genre)
        output_genre_path = os.path.join(WAVEFORM_PATH, genre)
        os.makedirs(output_genre_path, exist_ok=True)

        files = [f for f in os.listdir(genre_path) if f.endswith('.wav') or f.endswith('.au')]

        print(f'🎵 Memproses {subset} - genre: {genre} ({len(files)} file)')
        for filename in tqdm(files):
            file_path = os.path.join(genre_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                fig, ax = plt.subplots(figsize=(8, 4))

                librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.85)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")

                for spine in ax.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1)

                ax.text(
                    0.5, 1.02, genre.upper(), transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=12
                )

                fig.tight_layout()
                output_file = os.path.join(
                    output_genre_path,
                    filename.replace('.wav', '.png').replace('.au', '.png')
                )
                plt.savefig(output_file, bbox_inches='tight', pad_inches=0.3, dpi=150)
                plt.close(fig)

            except Exception as e:
                print(f'❌ Gagal memproses {file_path}: {e}')