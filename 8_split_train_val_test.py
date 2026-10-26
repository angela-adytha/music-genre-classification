import os
import shutil
from sklearn.model_selection import train_test_split

SEED = 42

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

DATASET_BASE = "fusi"
OUTPUT_BASE = "split"

def split_files(files, subset):
    """_summary_

    Args:
        files (_type_): _description_
        subset (_type_): _description_

    Returns:
        _type_: _description_
    """
    if subset == "sekunder":
        train_files, temp_files = train_test_split(
            files, test_size=0.3, random_state=SEED
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=1 / 3, random_state=SEED
        )
    else:
        n = len(files)
        n_train = round(0.1 * n)
        n_val = round(0.2 * n)
        train_files, temp_files = train_test_split(
            files, train_size=n_train, random_state=SEED
        )
        val_files, test_files = train_test_split(
            temp_files, train_size=n_val, random_state=SEED
        )
    return train_files, val_files, test_files


def copy_files(file_list, src_folder, dest_folder):
    """_summary_

    Args:
        file_list (_type_): _description_
        src_folder (_type_): _description_
        dest_folder (_type_): _description_
    """
    os.makedirs(dest_folder, exist_ok=True)
    for fname in file_list:
        shutil.copy(os.path.join(src_folder, fname), os.path.join(dest_folder, fname))


for method in METHODS:
    for data_subset in SUBSETS:
        dataset_path = os.path.join(DATASET_BASE, method, data_subset)
        output_path = os.path.join(OUTPUT_BASE, method, data_subset)

        print(f"\n===== Memproses: {method} / {data_subset} =====")

        if not os.path.isdir(dataset_path):
            print(f"⚠️  Lewati '{method}/{data_subset}': '{dataset_path}' tidak ditemukan.")
            continue

        if os.path.isdir(output_path) and os.listdir(output_path):
            print(f"⚠️  Lewati '{method}/{data_subset}': '{output_path}' sudah berisi data.")
            continue

        genres = sorted(
            g for g in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, g))
        )

        totals = {"train": 0, "validation": 0, "test": 0}

        for genre in genres:
            genre_path = os.path.join(dataset_path, genre)
            files = sorted(f for f in os.listdir(genre_path) if f.endswith(".png"))

            train_files, val_files, test_files = split_files(files, data_subset)

            assert not (set(train_files) & set(val_files)
                        | set(train_files) & set(test_files)
                        | set(val_files) & set(test_files))

            copy_files(train_files, genre_path, os.path.join(output_path, "train", genre))
            copy_files(val_files, genre_path, os.path.join(output_path, "validation", genre))
            copy_files(test_files, genre_path, os.path.join(output_path, "test", genre))

            totals["train"] += len(train_files)
            totals["validation"] += len(val_files)
            totals["test"] += len(test_files)

            print(f"Genre {genre}: Train={len(train_files)}, "
                  f"Val={len(val_files)}, Test={len(test_files)}")

        print(f"✅ {method}/{data_subset}: Train={totals['train']}, "
              f"Val={totals['validation']}, Test={totals['test']}")

print("\n✅ Stratified split selesai!")