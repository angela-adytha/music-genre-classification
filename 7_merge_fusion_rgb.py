import cv2
import os
import numpy as np

SUBSETS = [
    "sekunder",
    "primer/0:30-1:00",
    "primer/1:00-1:30",
    "primer/1:30-2:00",
    "primer/2:00-2:30",
    "primer/2:30-3:00",
    "primer/reff",
]

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]

METHODS = [
    "addition",
    "average",
    "division",
    "maximum",
    "minimum",
    "multiplication",
    "subtraction",
] + [f"weighted/{alpha}" for alpha in ALPHAS]

R_BASE = 'rgb/scalograms'
G_BASE = 'rgb/spectrograms'
B_BASE = 'rgb/fusi'

OUTPUT_BASE = 'fusi'

def normalize_channel(channel):
    """
    Menormalkan nilai piksel pada sebuah kanal ke rentang 0-255.
    
    Args:
        channel (np.array): Array NumPy yang merepresentasikan kanal grayscale.
    
    Returns:
        np.array: Kanal yang telah dinormalisasi dalam format uint8.
    """
    min_val = np.min(channel)
    max_val = np.max(channel)
    
    if max_val > min_val:
        normalized_channel = (channel - min_val) / (max_val - min_val) * 255
    else:
        normalized_channel = channel
        
    return normalized_channel.astype(np.uint8)

def process(R_dir, G_dir, B_dir, output_dir):
    """_summary_

    Args:
        R_dir (_type_): _description_
        G_dir (_type_): _description_
        B_dir (_type_): _description_
        output_dir (_type_): _description_
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    genres = os.listdir(R_dir)

    for genre in genres:
        genre_R_dir = os.path.join(R_dir, genre)
        genre_G_dir = os.path.join(G_dir, genre)
        genre_B_dir = os.path.join(B_dir, genre)
        genre_output_dir = os.path.join(output_dir, genre)

        if not os.path.isdir(genre_R_dir):
            continue

        if not os.path.exists(genre_output_dir):
            os.makedirs(genre_output_dir)

        file_list = sorted(os.listdir(genre_R_dir))

        for filename in file_list:
            R_path = os.path.join(genre_R_dir, filename)
            G_path = os.path.join(genre_G_dir, filename)
            B_path = os.path.join(genre_B_dir, filename)

            if os.path.exists(R_path) and os.path.exists(G_path) and os.path.exists(B_path):
                img_R = cv2.imread(R_path, cv2.IMREAD_GRAYSCALE)
                img_G = cv2.imread(G_path, cv2.IMREAD_GRAYSCALE)
                img_B = cv2.imread(B_path, cv2.IMREAD_GRAYSCALE)

                if img_R is None or img_G is None or img_B is None:
                    print(f"Gagal membaca citra: {os.path.join(genre, filename)}")
                    continue
                
                normalized_R = normalize_channel(img_R)
                normalized_G = normalize_channel(img_G)
                normalized_B = normalize_channel(img_B)

                fused_image = cv2.merge([normalized_B, normalized_G, normalized_R])

                output_path = os.path.join(genre_output_dir, filename)
                cv2.imwrite(output_path, fused_image)

                print(f"Citra fusi ternormalisasi {os.path.join(genre, filename)} berhasil disimpan.")

for method in METHODS:
    print(f"\n########## METHOD = {method} ##########")

    for subset in SUBSETS:
        R_dir = os.path.join(R_BASE, subset)
        G_dir = os.path.join(G_BASE, subset)
        B_dir = os.path.join(B_BASE, method, subset)
        output_dir = os.path.join(OUTPUT_BASE, method, subset)

        print(f"\n===== Memproses: {method} / {subset} =====")

        if not os.path.isdir(R_dir) or not os.path.isdir(G_dir) or not os.path.isdir(B_dir):
            print(f"⚠️  Lewati '{method}/{subset}': salah satu folder input tidak ditemukan.")
            continue

        process(R_dir, G_dir, B_dir, output_dir)

print("\nProses fusi dengan normalisasi selesai.")