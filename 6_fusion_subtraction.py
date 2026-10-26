import os
import cv2
import numpy as np

SCALOGRAM_BASE = "rgb/scalograms"
SPECTROGRAM_BASE = "rgb/spectrograms"
OUTPUT_BASE = "rgb/fusi/subtraction"

SUBSETS = [
    "sekunder",
    "primer/0:30-1:00",
    "primer/1:00-1:30",
    "primer/1:30-2:00",
    "primer/2:00-2:30",
    "primer/2:30-3:00",
    "primer/reff",
]

def is_image_file(name: str) -> bool:
    """_summary_

    Args:
        name (str): _description_

    Returns:
        bool: _description_
    """
    name = name.lower()
    return name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))

def ensure_dir(path: str):
    """_summary_

    Args:
        path (str): _description_
    """
    os.makedirs(path, exist_ok=True)

def process_subset(scalogram_dir: str, spectrogram_dir:str, output_dir: str):
    """_summary_
    """
    ensure_dir(output_dir)

    genres_scal = set(os.listdir(scalogram_dir))
    genres_spec = set(os.listdir(spectrogram_dir))
    all_genres = sorted(genres_scal | genres_spec)

    for genre in all_genres:
        scal_genre_path = os.path.join(scalogram_dir, genre)
        spec_genre_path = os.path.join(spectrogram_dir, genre)

        if not os.path.isdir(scal_genre_path):
            print(f"⚠️  Lewati genre '{genre}' karena tidak ada di '{scalogram_dir}'.")
            continue
        if not os.path.isdir(spec_genre_path):
            print(f"⚠️  Lewati genre '{genre}' karena tidak ada di '{spectrogram_dir}'.")
            continue

        out_genre_path = os.path.join(output_dir, genre)
        ensure_dir(out_genre_path)

        for fname in os.listdir(scal_genre_path):
            if not is_image_file(fname):
                continue

            scal_path = os.path.join(scal_genre_path, fname)
            spec_path = os.path.join(spec_genre_path, fname)

            if not os.path.exists(spec_path):
                print(f"❌ Pasangan untuk '{fname}' di genre '{genre}' tidak ditemukan di spectrograms. Lewati.")
                continue

            scal_img = cv2.imread(scal_path, cv2.IMREAD_COLOR)
            spec_img = cv2.imread(spec_path, cv2.IMREAD_COLOR)

            if scal_img is None or spec_img is None:
                print(f"❌ Gagal membaca file: {fname}")
                continue

            if scal_img.shape[:2] != spec_img.shape[:2]:
                spec_img = cv2.resize(
                    spec_img,
                    (scal_img.shape[1], scal_img.shape[0]),
                    interpolation=cv2.INTER_AREA
                )

            R_from_scal = scal_img[:, :, 2]
            G_from_spec = spec_img[:, :, 1]

            B_fused = cv2.subtract(R_from_scal, G_from_spec)

            zeros = np.zeros_like(B_fused, dtype=B_fused.dtype)
            blue_only = cv2.merge([B_fused, zeros, zeros])

            out_path = os.path.join(out_genre_path, fname)
            cv2.imwrite(out_path, blue_only)
            print(f"Simpan: {out_path}")

    print(f"🎯 Selesai. Hasil fusi tersimpan di: {output_dir}")

def main():
    for subset in SUBSETS:
        scalogram_dir = os.path.join(SCALOGRAM_BASE, subset)
        spectrogram_dir = os.path.join(SPECTROGRAM_BASE, subset)
        output_dir = os.path.join(OUTPUT_BASE, subset)

        print(f"\n===== Memproses: {subset} =====")

        if not os.path.isdir(scalogram_dir):
            print(f"⚠️  Lewati subset '{subset}': '{scalogram_dir}' tidak ditemukan.")
            continue
        if not os.path.isdir(spectrogram_dir):
            print(f"⚠️  Lewati subset '{subset}': '{spectrogram_dir}' tidak ditemukan.")
            continue

        process_subset(scalogram_dir, spectrogram_dir, output_dir)

    print(f"\n🎯 Selesai. Hasil fusi tersimpan di: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()