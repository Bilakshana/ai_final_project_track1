# Track 1 — Waste Image Classification
### Custom CNN vs MobileNetV2 on TrashNet

A complete end-to-end image classification project comparing a hand-built CNN
with a fine-tuned MobileNetV2 model on the 6-class TrashNet dataset.

---

## Project Structure

```
track1_project/
├── preprocess.py          # Data loading, augmentation, train/val split
├── models.py              # Model definitions (Custom CNN + MobileNetV2)
├── train.py               # Training loop, callbacks, plot saving
├── evaluate.py            # Metrics, confusion matrices, error analysis
├── app.py                 # Streamlit web demo
├── download_dataset.py    # Automated dataset downloader
├── requirements.txt       # Python dependencies
│
├── data/                  # (created after split)
│   ├── train/
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   └── val/
│       └── …
│
├── models/                # (created after training)
│   ├── custom_cnn.h5
│   ├── mobilenetv2.h5
│   ├── custom_cnn_history.json
│   └── mobilenetv2_history.json
│
└── plots/                 # (created after training/evaluation)
    ├── custom_cnn_history.png
    ├── mobilenetv2_history.png
    ├── confusion_matrix_custom_cnn.png
    ├── confusion_matrix_mobilenetv2.png
    ├── comparison_table.png
    └── error_analysis_*.png
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Get the TrashNet dataset

**Option A — Kaggle API (easiest)**
```bash
pip install kaggle
# Put kaggle.json in ~/.kaggle/   (get it from kaggle.com → Settings → API)
python download_dataset.py --source kaggle --split
```

**Option B — Browser download**
```bash
python download_dataset.py --source manual
# Follow the link, download the zip, then:
python download_dataset.py --source local --zip ~/Downloads/trashnet.zip --split
```

**Option C — Already have the data**
```bash
# Make sure you have:  dataset/cardboard/*.jpg  dataset/glass/*.jpg  …etc.
python -c "from preprocess import split_dataset; split_dataset('dataset/', 'data/')"
```

### 3 — Train both models

```bash
python train.py
```

> ⏱ GPU: ~30–60 min total  
> ⏱ CPU: ~3–5 hours (coffee recommended ☕)

### 4 — Evaluate and compare

```bash
python evaluate.py
```

Prints per-class metrics, saves confusion matrices and error analysis plots.

### 5 — Run the web demo

```bash
streamlit run app.py
```

Open http://localhost:8501 — upload any waste image, get a prediction + tip.

---

## Architecture Details

### Model 1 — Custom CNN (Baseline)

```
Input (224×224×3)
  ↓
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)   # Block 1
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.30)   # Block 2
Conv2D(128)→ BatchNorm → MaxPool → Dropout(0.35)   # Block 3
  ↓
GlobalAveragePooling2D
Dense(256, ReLU) → Dropout(0.50)
Dense(6, Softmax)
```

- ~1.4M parameters, all trained from scratch
- L2 regularisation on all Conv and Dense layers
- GlobalAvgPool prevents overfitting on small dataset

### Model 2 — MobileNetV2 (Transfer Learning)

```
Input (224×224×3)
  ↓
MobileNetV2 base (ImageNet weights)
  │  Phase 1: ALL FROZEN  →  only head trains
  │  Phase 2: top 54 layers unfrozen  →  fine-tune with LR=1e-5
  ↓
GlobalAveragePooling2D
Dense(256, ReLU) → Dropout(0.40)
Dense(6, Softmax)
```

- ~2.3M params in base; only ~330K in new head
- Phase 2 fine-tuning avoids catastrophic forgetting
- BatchNorm layers stay frozen in Phase 2 (ImageNet stats preserved)

---

## Expected Results

| Metric    | Custom CNN  | MobileNetV2 |
|-----------|-------------|-------------|
| Accuracy  | 72–76%      | 88–92%      |
| F1-Score  | 71–74%      | 87–91%      |
| Precision | 70–75%      | 87–90%      |
| Recall    | 72–76%      | 88–92%      |

**Why does MobileNetV2 win so decisively?**

MobileNetV2 was pre-trained on 1.28 million ImageNet images. It already
"knows" low-level features (edges, curves, textures) and mid-level ones
(surfaces, reflectivity, shape). Fine-tuning on TrashNet just teaches it
the final waste-specific mapping. The Custom CNN has to learn *everything*
from only ~2,000 training images — statistically hard.

---

## Common Confusion Pairs (Error Analysis)

| True Class  | Predicted   | Reason |
|-------------|-------------|--------|
| Glass       | Metal       | Both have shiny/reflective surfaces |
| Paper       | Cardboard   | Similar colour, texture, and flat shape |
| Plastic     | Glass       | Transparent plastic bags ≈ clear glass bottles |
| Trash       | Paper       | Catch-all "trash" includes crumpled paper items |

These confusions persist even in MobileNetV2 — they are *visual* ambiguities
inherent in 2-D photography, not model weaknesses.

---

## Augmentation Strategy

| Transform         | Training | Validation |
|-------------------|----------|------------|
| Rescale (÷255)    | ✅       | ✅         |
| Rotation (±20°)   | ✅       | ❌         |
| Horizontal flip   | ✅       | ❌         |
| Zoom (±20%)       | ✅       | ❌         |
| Width/height shift| ✅       | ❌         |
| Shear             | ✅       | ❌         |

Augmentation is applied *only* at training time. Validation data uses
raw (rescaled) images to simulate real-world deployment conditions.

---

## Callbacks Used

| Callback            | Purpose |
|---------------------|---------|
| `EarlyStopping`     | Stop when val_accuracy plateaus (patience=10/8) |
| `ModelCheckpoint`   | Save the single best model checkpoint |
| `ReduceLROnPlateau` | Halve LR after 5 epochs of no val_loss improvement |
| `CSVLogger`         | Save per-epoch metrics to CSV for later analysis |

---

## Dataset

**TrashNet** — Gary Thung & Mindy Yang (2016)  
2,527 images across 6 classes:

| Class     | Count |
|-----------|-------|
| cardboard | 403   |
| glass     | 501   |
| metal     | 410   |
| paper     | 594   |
| plastic   | 482   |
| trash     | 137   |

Source: https://github.com/garythung/trashnet  
Kaggle mirror: https://www.kaggle.com/datasets/feyzazkefe/trashnet

---

## Tips for Better Results

1. **More data** — Supplement TrashNet with web-scraped images or the
   Open Images dataset (waste subset).

2. **Test-time augmentation (TTA)** — Predict on 5–10 augmented versions
   of each test image and average probabilities.

3. **Ensemble** — Combine CNN + MobileNetV2 predictions (weighted average).
   Typically adds 1–3% accuracy with no retraining.

4. **Better backbone** — Try EfficientNetB0 (smaller, usually +2–4% over
   MobileNetV2) or ConvNeXt-Tiny.

5. **Class weights** — TrashNet's "trash" class has only 137 images.
   Use `class_weight` in `model.fit()` to penalise misses more.
