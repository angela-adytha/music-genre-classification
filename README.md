
# **Application of a Combination of ResNet and Bi-GRU Methods for Scalogram-Spectrogram Image Fusion-Based Music Genre Classification**

## 1. Overview

Audio signals are converted into two time-frequency images:

- **Scalogram** from CWT (Morlet wavelet), mapped to the **Red** channel.
- **Spectrogram** from STFT, mapped to the **Green** channel.
- **Fusion result** of the two (addition, average, division, maximum, minimum, multiplication, subtraction, or weighted average), mapped to the **Blue** channel.

The fused 224×224 RGB image is the input to a **ResNet18 + Bi-GRU** model with 10 genre classes:
`blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`.

Configuration reported in the notebook: *weighted average* fusion with α = 0.5.

## 2. Repository Structure

```
├── 1_generate_waveforms.py
├── 2_normalize_audio.py
├── 3_generate_scalogram_spectrogram.py
├── 4_crop_image_segments.py
├── 5_colorize_rgb_channels.py
├── 6_fusion_addition.py
├── 6_fusion_average.py
├── 6_fusion_division.py
├── 6_fusion_maximum.py
├── 6_fusion_minimum.py
├── 6_fusion_multiplication.py
├── 6_fusion_subtraction.py
├── 6_fusion_weighted_average.py
├── 7_merge_fusion_rgb.py
├── 8_split_train_val_test.py
├── 9_resgru_training_and_prediction.ipynb
├── requirements.txt
└── README.md
```

## 3. Pipeline

Run from the repo root, **in order**. Each script reads the output of the previous stage.

| Script                                     | Input                   | Output                                                 | Function                                                               |
| ------------------------------------------ | ----------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------- |
| `1_generate_waveforms.py`                | `dataset/raw/`        | `waveforms/`                                         | Waveform visualization per file                                        |
| `2_normalize_audio.py`                   | `dataset/raw/`        | `dataset/normal/`                                    | Amplitude normalization (peak to ±1)                                  |
| `3_generate_scalogram_spectrogram.py`    | `dataset/normal/`     | `grayscale/spectrograms/`, `grayscale/scalograms/` | STFT and CWT, grayscale 2240×224 (30 seconds)                         |
| `4_crop_image_segments.py`               | `grayscale/`          | `crop/`                                              | Split into 10 segments of 224×224 (≈3 seconds each)                  |
| `5_colorize_rgb_channels.py`             | `crop/`               | `rgb/scalograms/`, `rgb/spectrograms/`             | Scalogram to red channel, spectrogram to green channel                 |
| `6_fusion_*.py`                          | `rgb/`                | `rgb/fusi/<method>/`                                 | Fuse the R and G channels into a single channel (B)                    |
| `7_merge_fusion_rgb.py`                  | `rgb/`, `rgb/fusi/` | `fusi/<method>/`                                     | Merge R, G, B into an RGB image (each channel min-max scaled to 0-255) |
| `8_split_train_val_test.py`              | `fusi/`               | `split/<method>/<subset>/{train,validation,test}/`   | Data split (see Data Split section)                                    |
| `9_resgru_training_and_prediction.ipynb` | `split/`              | `checkpoints/`, `plot/`, `confusion_matrix/`     | Training, evaluation, and prediction                                   |

Fusion method notes:

- Script 6 produces all methods. `6_fusion_weighted_average.py` runs α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, with weight α on the scalogram and (1−α) on the spectrogram.

## 4. Data

| Subset         | Contents                                       | Link                                                                                                                                                                                    |
| -------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sekunder`   | GTZAN, 10 genres × 100 files, 30-second clips | [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)       |
| `primer/...` | 50 songs (5 per genre)                         | [https://youtube.com/playlist?list=PL5NgJWatxKORrTwUGs-ZwqMWtQaxEGqRS&amp;si=dabpFmt3mf_JF5hS](https://youtube.com/playlist?list=PL5NgJWatxKORrTwUGs-ZwqMWtQaxEGqRS&si=dabpFmt3mf_JF5hS) |

The primary (`primer`) subset consists of 30-second excerpts from each song: `0:30-1:00`, `1:00-1:30`, `1:30-2:00`, `2:00-2:30`, `2:30-3:00`, and `reff` (the chorus).

Input folder structure expected by the scripts:

```
dataset/raw/
├── sekunder/<genre>/*.wav
└── primer/
    ├── 0:30-1:00/<genre>/*.wav
    ├── 1:00-1:30/<genre>/*.wav
    ├── 1:30-2:00/<genre>/*.wav
    ├── 2:00-2:30/<genre>/*.wav
    ├── 2:30-3:00/<genre>/*.wav
    └── reff/<genre>/*.wav
```

Notes:

- The normalization (2) and feature extraction (3) scripts only process **`.wav`** files. Convert `.au` to `.wav` first if needed.
- Corrupted or too-short audio files were removed manually before the pipeline (9,980 of 10,000 GTZAN images are used). List of removed files: **[TO FILL]**.
- The folder name `0:30-1:00` contains a `:` character, which is **not valid on Windows**. Run on Linux/macOS/WSL, or rename it in all scripts.
- The script that cuts the primary songs into excerpts from the original source is **not included**.

## 5. Data Split

`8_split_train_val_test.py` splits the data per genre with `random_state=42`:

| Subset       | Train | Validation | Test |
| ------------ | ----- | ---------- | ---- |
| `sekunder` | 70%   | 20%        | 10%  |
| `primer/*` | 10%   | 20%        | 70%  |

Number of images in the notebook (α=0.5 weighting, 7 sources combined):

| Split      | Secondary | Primary  | Total |
| ---------- | --------- | -------- | ----- |
| Train      | 6,986     | 6 × 50  | 7,286 |
| Validation | 1,996     | 6 × 100 | 2,596 |
| Test       | 998       | 6 × 350 | 3,098 |

## 6. Model

Defined in the notebook `9_resgru_training_and_prediction.ipynb`:

1. ResNet18 pretrained on ImageNet without `avgpool` and `fc`: 224×224×3 input becomes 512×7×7.
2. `AdaptiveMaxPool2d(4×4)` + `BatchNorm2d(512)` + `Dropout(0.1)`, three times in sequence.
3. Reshape into 16 steps × 512 features.
4. 2-layer Bi-GRU, hidden size 256. The output of the last step is used.
5. ReLU, then `Linear(512 → 10)`.

13,550,154 parameters in total, all trainable.

Training:

| Parameter            | Value                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------- |
| Optimizer            | Adam, lr 1e-4, weight decay 1e-4                                                        |
| Loss                 | CrossEntropyLoss                                                                        |
| Batch size           | 16                                                                                      |
| Maximum epochs       | 70                                                                                      |
| Scheduler            | ReduceLROnPlateau (val loss, factor 0.5, patience 10, min lr 1e-7)                      |
| Early stopping       | patience 30 (val loss)                                                                  |
| Checkpoint           | `checkpoints/best_model.pth` whenever val loss improves                               |
| Augmentation (train) | HorizontalFlip, Rotation 10°, RandomResizedCrop (0.9–1.0), ColorJitter (contrast 0.1) |

## 7. How to Run

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Place the audio in dataset/raw/ following the structure above

# 3. Data pipeline
python 1_generate_waveforms.py          # optional (visualization)
python 2_normalize_audio.py
python 3_generate_scalogram_spectrogram.py
python 4_crop_image_segments.py
python 5_colorize_rgb_channels.py
python 6_fusion_weighted_average.py     # or another 6_fusion_* script
python 7_merge_fusion_rgb.py
python 8_split_train_val_test.py

# 4. Training and evaluation
jupyter notebook 9_resgru_training_and_prediction.ipynb
```

Hardware requirements:

- A CUDA GPU is strongly recommended for the notebook. Without a GPU, the notebook falls back to CPU and is very slow.
- Large disk space. Each fusion method produces tens of thousands of PNGs across several stages.
- Script 3 (CWT on 30 seconds of audio) uses a fairly large amount of memory per file.

## 8. Notebook Results

Accuracy on the seven test sets (single run, α = 0.5, final training weights):

| Test set              | Number of images | Accuracy |
| --------------------- | ---------------- | -------- |
| 1 (secondary)         | 998              | 0.9529   |
| 2 (primary 0:30-1:00) | 350              | 0.9200   |
| 3 (primary 1:00-1:30) | 350              | 0.9200   |
| 4 (primary 1:30-2:00) | 350              | 0.9000   |
| 5 (primary 2:00-2:30) | 350              | 0.9286   |
| 6 (primary 2:30-3:00) | 350              | 0.9229   |
| 7 (primary reff)      | 350              | 0.9571   |

The `rock` genre has the lowest recall in almost all test sets. Best epoch: 68 with val accuracy 0.9588 and val loss 0.1721.

## 9. Limitations

- Splitting the data per crop can inflate accuracy.
- The global PyTorch seed is not set, so training results are not fully reproducible.
- Small sample size: 50 primary songs, 10 songs per genre in the per-song scenario.

## 10. Contact

* **Name:** Angela Adytha Putri
* **Email:** angel.adytha@gmail.com
* **GitHub:** https://github.com/angela-adytha
* **LinkedIn:** https://www.linkedin.com/in/angela-adytha-putri-a8b6082b9/
