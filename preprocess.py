"""
preprocess.py — Data loading, augmentation, and dataset splitting for Track 1.
Updated for TensorFlow 2.20+ / Keras 3 (no ImageDataGenerator).
"""

import os
import shutil
import random
from pathlib import Path

import tensorflow as tf

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES    = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE   = 224
BATCH_SIZE = 32

# Augmentation layer (applied only during training)
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.10, 0.10),
], name="augmentation")


# ── Generator factory ──────────────────────────────────────────────────────────
def get_generators(data_dir: str = "data/",
                   img_size: int = IMG_SIZE,
                   batch_size: int = BATCH_SIZE):
    """
    Returns (train_dataset, val_dataset) as tf.data.Dataset objects.
    Drop-in replacement for the old ImageDataGenerator flow_from_directory.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    # Normalise to [0, 1] and apply augmentation on train only
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # Attach class info so train.py / evaluate.py can read it
    train_ds.class_names = CLASSES
    val_ds.class_names   = CLASSES

    print(f"\n✅ Datasets ready")
    print(f"   Classes : {CLASSES}")
    return train_ds, val_ds


# ── Dataset splitter ───────────────────────────────────────────────────────────
def split_dataset(src_dir: str,
                  dst_dir: str,
                  val_split: float = 0.20,
                  seed: int = 42):
    """
    Splits flat TrashNet folder into train/ and val/ inside dst_dir.
    """
    random.seed(seed)
    src = Path(src_dir)
    dst = Path(dst_dir)

    for cls in CLASSES:
        cls_src = src / cls
        if not cls_src.exists():
            print(f"⚠️  Skipping '{cls}' — not found in {src_dir}")
            continue

        images = (list(cls_src.glob("*.jpg")) +
                  list(cls_src.glob("*.jpeg")) +
                  list(cls_src.glob("*.png")))
        random.shuffle(images)

        n_val        = int(len(images) * val_split)
        val_images   = images[:n_val]
        train_images = images[n_val:]

        for split, imgs in [("train", train_images), ("val", val_images)]:
            out_dir = dst / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy2(img_path, out_dir / img_path.name)

        print(f"  {cls:12s}: {len(train_images)} train  |  {len(val_images)} val")

    print(f"\n✅ Dataset split complete → {dst_dir}")


# ── Quick sanity-check ─────────────────────────────────────────────────────────
def dataset_stats(data_dir: str = "data/"):
    for split in ("train", "val"):
        print(f"\n── {split.upper()} ──")
        total = 0
        for cls in CLASSES:
            p = Path(data_dir) / split / cls
            n = len(list(p.glob("*.*"))) if p.exists() else 0
            print(f"  {cls:12s}: {n}")
            total += n
        print(f"  {'TOTAL':12s}: {total}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if Path("data/train").exists():
        dataset_stats("data/")
        print("\nTesting dataset loading…")
        train_ds, val_ds = get_generators("data/")
        for imgs, labels in train_ds.take(1):
            print(f"  Batch shape : {imgs.shape}")
            print(f"  Label shape : {labels.shape}")
            print(f"  Pixel range : [{imgs.numpy().min():.2f}, {imgs.numpy().max():.2f}]")
        print("✅ All good — run: python train.py")
    else:
        print(
            "\ndata/train not found.\n"
            "Run the dataset download + split first:\n"
            "  python download_dataset.py --source manual\n"
        )
