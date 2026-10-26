import os
import librosa
import soundfile as sf

SUBSETS = [
    "sekunder",
    "primer/0:30-1:00",
    "primer/1:00-1:30",
    "primer/1:30-2:00",
    "primer/2:00-2:30",
    "primer/2:30-3:00",
    "primer/reff",
]

def normalize(input_folder, output_folder):
    """
    Normalizes the amplitude of all audio files in a given folder structure
    and saves them to a new output folder.

    Args:
        input_folder (str): Path to the folder containing raw audio files.
        output_folder (str): Path to the folder where normalized files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    for genre_folder in os.listdir(input_folder):
        genre_path = os.path.join(input_folder, genre_folder)
        output_genre_path = os.path.join(output_folder, genre_folder)
        os.makedirs(output_genre_path, exist_ok=True)

        if os.path.isdir(genre_path):
            print(f"Processing genre: {genre_folder}")
            for filename in os.listdir(genre_path):
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    input_filepath = os.path.join(genre_path, filename)
                    output_filepath = os.path.join(output_genre_path, filename)

                    try:
                        y, sr = librosa.load(input_filepath, sr=None, mono=True)

                        y_normalized = librosa.util.normalize(y)

                        sf.write(output_filepath, y_normalized, sr)
                        print(f"Normalized and saved: {output_filepath}")

                    except Exception as e:
                        print(f"Error processing {input_filepath}: {e}")

if __name__ == "__main__":
    for subset in SUBSETS:
        input_data_folder = os.path.join("dataset/raw", subset)
        output_normalized_folder = os.path.join("dataset/normal", subset)

        print(f"\n===== Memproses: {subset} =====")
        normalize(input_data_folder, output_normalized_folder)
        print("Audio normalization complete!")
        print(f"Normalized files are saved in: {output_normalized_folder}")