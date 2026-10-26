import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pywt
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

DATASET_BASE = 'dataset/normal'
SPECTROGRAM_BASE_OUTPUT_PATH = 'grayscale/spectrograms'
SCALOGRAM_BASE_OUTPUT_PATH = 'grayscale/scalograms'

SUBSETS = [
    'sekunder',
    'primer/0:30-1:00',
    'primer/1:00-1:30',
    'primer/1:30-2:00',
    'primer/2:00-2:30',
    'primer/2:30-3:00',
    'primer/reff',
]

IMG_WIDTH = 2240
IMG_HEIGHT = 224

def spectrogram(audio_path, genre, root_output_path, filename):
    """_summary_

    Args:
        audio_path (_type_): _description_
        genre (_type_): _description_
        root_output_path (_type_): _description_
        filename (_type_): _description_
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=30)
        D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
        S_power = np.abs(D)**2
        S_db = librosa.power_to_db(S_power, ref=np.max)

        genre_output_path = os.path.join(root_output_path, genre)
        os.makedirs(genre_output_path, exist_ok=True)

        plt.figure(figsize=(IMG_WIDTH/100, IMG_HEIGHT/100), dpi=100)
        librosa.display.specshow(S_db, sr=sr, cmap='gray_r')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.margins(0,0)

        output_filepath = os.path.join(genre_output_path, f"{os.path.splitext(filename)[0]}.png")
        plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        print(f"Spectrogram saved: {output_filepath}")

    except Exception as e:
        print(f"Error processing {audio_path} for spectrogram: {e}")

def scalogram(audio_path, genre, root_output_path, filename):
    """_summary_

    Args:
        audio_path (_type_): _description_
        genre (_type_): _description_
        root_output_path (_type_): _description_
        filename (_type_): _description_
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=30)
        widths = np.arange(1, 64)

        try:
            cwtmatr, freqs = pywt.cwt(y, widths, 'morl', sampling_period=1/sr)
        except ValueError as e:
            print(f"Skipping scalogram for {audio_path} due to CWT error: {e}")
            return

        cwt_power = np.abs(cwtmatr)**2
        cwt_db = librosa.power_to_db(cwt_power, ref=np.max)

        genre_output_path = os.path.join(root_output_path, genre)
        os.makedirs(genre_output_path, exist_ok=True)

        plt.figure(figsize=(IMG_WIDTH/100, IMG_HEIGHT/100), dpi=100)
        plt.imshow(cwt_db, cmap='gray_r', aspect='auto', origin='lower')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.margins(0,0)

        output_filepath = os.path.join(genre_output_path, f"{os.path.splitext(filename)[0]}.png")
        plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        print(f"Scalogram saved: {output_filepath}")

    except Exception as e:
        print(f"Error processing {audio_path} for scalogram: {e}")

def main():
    """_summary_
    """
    for subset in SUBSETS:
        dataset_path = os.path.join(DATASET_BASE, subset)
        spectrogram_output_path = os.path.join(SPECTROGRAM_BASE_OUTPUT_PATH, subset)
        scalogram_output_path = os.path.join(SCALOGRAM_BASE_OUTPUT_PATH, subset)

        print(f"\n===== Memproses: {subset} =====")
        print(f"Processing audio files from: {dataset_path}")
        for root, dirs, files in os.walk(dataset_path):
            if root == dataset_path:
                continue

            genre = os.path.basename(root)
            print(f"\nProcessing genre: {genre}")

            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    print(f"  Processing: {audio_path}")

                    spectrogram(audio_path, genre, spectrogram_output_path, file)
                    scalogram(audio_path, genre, scalogram_output_path, file)
    print("\nAll audio files processed.")

if __name__ == "__main__":
    main()