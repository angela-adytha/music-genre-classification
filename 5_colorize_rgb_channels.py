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

konfigurasi = []
for subset in SUBSETS:
    konfigurasi.append({
        "input": f"crop/scalograms/{subset}",
        "output": f"rgb/scalograms/{subset}",
        "warna": "merah"
    })
    konfigurasi.append({
        "input": f"crop/spectrograms/{subset}",
        "output": f"rgb/spectrograms/{subset}",
        "warna": "hijau"
    })

for config in konfigurasi:
    folder_input = config["input"]
    folder_output = config["output"]
    warna = config["warna"]

    print(f"\nMemproses folder '{folder_input}' → '{folder_output}' ({warna})")

    for dirpath, dirnames, filenames in os.walk(folder_input):
        relative_path = os.path.relpath(dirpath, folder_input)
        target_dirpath = os.path.join(folder_output, relative_path)
        os.makedirs(target_dirpath, exist_ok=True)

        for nama_file in filenames:
            if nama_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path_file_sumber = os.path.join(dirpath, nama_file)
                path_file_tujuan = os.path.join(target_dirpath, nama_file)

                try:
                    grayscale_img = Image.open(path_file_sumber).convert('L')

                    if warna == "merah":
                        red = grayscale_img
                        green = Image.new('L', grayscale_img.size, 0)
                        blue = Image.new('L', grayscale_img.size, 0)
                    elif warna == "hijau":
                        red = Image.new('L', grayscale_img.size, 0)
                        green = grayscale_img
                        blue = Image.new('L', grayscale_img.size, 0)

                    rgb_img = Image.merge('RGB', (red, green, blue))
                    rgb_img.save(path_file_tujuan)

                    print(f"  ✔ '{nama_file}' disimpan di '{path_file_tujuan}'")

                except Exception as e:
                    print(f"  ✖ Gagal memproses '{path_file_sumber}': {e}")

print("\n✅ Semua file telah selesai diproses.")