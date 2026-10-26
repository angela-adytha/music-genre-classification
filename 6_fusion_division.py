import os
import cv2
import numpy as np

SCALOGRAM_BASE = "rgb/scalograms"
SPECTROGRAM_BASE = "rgb/spectrograms"
OUTPUT_BASE = "rgb/fusi/division"

SUBSETS = [
    "sekunder",
    "primer/0:30-1:00",
    "primer/1:00-1:30",
    "primer/1:30-2:00",
    "primer/2:00-2:30",
    "primer/2:30-3:00",
    "primer/reff",
]

print("Memulai proses fusi citra...")

for subset in SUBSETS:
    SCALOGRAM_DIR = os.path.join(SCALOGRAM_BASE, subset)
    SPECTROGRAM_DIR = os.path.join(SPECTROGRAM_BASE, subset)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, subset)

    print(f"\n===== Memproses: {subset} =====")

    if not os.path.isdir(SCALOGRAM_DIR):
        print(f"⚠️  Lewati subset '{subset}': '{SCALOGRAM_DIR}' tidak ditemukan.")
        continue
    if not os.path.isdir(SPECTROGRAM_DIR):
        print(f"⚠️  Lewati subset '{subset}': '{SPECTROGRAM_DIR}' tidak ditemukan.")
        continue

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    genres_scalo = {d for d in os.listdir(SCALOGRAM_DIR) if os.path.isdir(os.path.join(SCALOGRAM_DIR, d))}
    genres_spectro = {d for d in os.listdir(SPECTROGRAM_DIR) if os.path.isdir(os.path.join(SPECTROGRAM_DIR, d))}

    genres = sorted(list(genres_scalo.intersection(genres_spectro)))

    for genre in genres:
        genre_output_dir = os.path.join(OUTPUT_DIR, genre)
        os.makedirs(genre_output_dir, exist_ok=True)

        scalogram_files = set(os.listdir(os.path.join(SCALOGRAM_DIR, genre)))
        spectrogram_files = set(os.listdir(os.path.join(SPECTROGRAM_DIR, genre)))
        image_files = sorted(list(scalogram_files.intersection(spectrogram_files)))

        for image_file in image_files:
            scalogram_path = os.path.join(SCALOGRAM_DIR, genre, image_file)
            spectrogram_path = os.path.join(SPECTROGRAM_DIR, genre, image_file)

            if os.path.exists(scalogram_path) and os.path.exists(spectrogram_path):
                scalogram = cv2.imread(scalogram_path)
                spectrogram = cv2.imread(spectrogram_path)

                r_channel = scalogram[:, :, 2]
                g_channel = spectrogram[:, :, 1] 

                b_fused = np.divide(r_channel.astype(np.float32), g_channel.astype(np.float32) + 1e-8) * 255.0

                b_fused = np.clip(b_fused, 0, 255).astype(np.uint8)

                fused_image = np.zeros_like(scalogram)
                fused_image[:, :, 0] = b_fused 
                
                out_path = os.path.join(genre_output_dir, image_file)
                cv2.imwrite(out_path, fused_image)
                print(f"Simpan: {out_path}")

print(f"🎯 Selesai. Hasil fusi tersimpan di: {OUTPUT_BASE}")