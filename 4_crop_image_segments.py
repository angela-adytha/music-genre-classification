import os
from PIL import Image

SUBSETS = [
    "sekunder",
    "primer/0:30-1:00",
    "primer/1:00-1:30",
    "primer/1:30-2:00",
    "primer/2:00-2:30",
    "primer/2:30-3:00",
    "primer/reff",
]

SPECTROGRAMS_BASE = "grayscale/spectrograms"
SCALOGRAMS_BASE   = "grayscale/scalograms"

SPECTROGRAMS_OUT_BASE = "crop/spectrograms"
SCALOGRAMS_OUT_BASE   = "crop/scalograms"

def crop(input_dir, output_dir):
    """_summary_

    Args:
        input_dir (_type_): _description_
        output_dir (_type_): _description_
    """
    os.makedirs(output_dir, exist_ok=True)

    for genre in os.listdir(input_dir):
        genre_path = os.path.join(input_dir, genre)
        if not os.path.isdir(genre_path):
            continue
        
        output_genre_path = os.path.join(output_dir, genre)
        os.makedirs(output_genre_path, exist_ok=True)

        for file in os.listdir(genre_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(genre_path, file)
            img = Image.open(img_path)

            w, h = img.size
            if (w, h) != (2240, 224):
                print(f"⚠️ Ukuran tidak sesuai: {img_path} ({w}x{h})")
                continue

            for i in range(10):
                left = i * 224
                upper = 0
                right = left + 224
                lower = 224

                cropped = img.crop((left, upper, right, lower))
                
                file_name, ext = os.path.splitext(file)
                cropped_name = f"{file_name}_part{i+1}{ext}"
                cropped.save(os.path.join(output_genre_path, cropped_name))

            print(f"✔ {file} dipotong jadi 10 gambar")

for subset in SUBSETS:
    print(f"\n===== Memproses: {subset} =====")
    crop(os.path.join(SPECTROGRAMS_BASE, subset), os.path.join(SPECTROGRAMS_OUT_BASE, subset))
    crop(os.path.join(SCALOGRAMS_BASE, subset), os.path.join(SCALOGRAMS_OUT_BASE, subset))

print("✅ Selesai memotong semua gambar.")