"""
train.py — Train both models and save results.

Run:
    python train.py

Outputs:
    models/custom_cnn.h5
    models/mobilenetv2.h5
    plots/custom_cnn_history.png
    plots/mobilenetv2_history.png
"""

import os
import time
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)

from preprocess import get_generators, CLASSES
from models import build_custom_cnn, build_mobilenetv2, unfreeze_for_finetuning


# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "data/"
MODEL_DIR  = "models/"
PLOT_DIR   = "plots/"
IMG_SIZE   = 224
BATCH_SIZE = 32

CNN_EPOCHS    = 60    # EarlyStopping will cut this short if needed
MOB_EPOCHS_P1 = 20   # Phase 1: head only
MOB_EPOCHS_P2 = 30   # Phase 2: fine-tuning

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ── Plotting helper ─────────────────────────────────────────────────────────────
def plot_history(history, title: str, save_path: str):
    """Save accuracy + loss curves side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history["accuracy"],     label="Train",      color="#2196F3")
    axes[0].plot(history["val_accuracy"], label="Validation", color="#FF5722")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history["loss"],     label="Train",      color="#2196F3")
    axes[1].plot(history["val_loss"], label="Validation", color="#FF5722")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📈 Plot saved → {save_path}")


def merge_histories(h1, h2):
    """Merge two Keras history dicts (Phase 1 + Phase 2)."""
    merged = {}
    for key in h1.keys():
        merged[key] = h1[key] + h2.get(key, [])
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Train Custom CNN
# ══════════════════════════════════════════════════════════════════════════════
def train_custom_cnn(train_gen, val_gen):
    print("\n" + "═" * 60)
    print("  Training  Custom CNN  (from scratch)")
    print("═" * 60)

    model = build_custom_cnn(num_classes=len(CLASSES), img_size=IMG_SIZE)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "custom_cnn_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(os.path.join(MODEL_DIR, "custom_cnn_log.csv")),
    ]

    t0 = time.time()
    hist = model.fit(
        train_gen,
        epochs=CNN_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0

    # Save final model
    out_path = os.path.join(MODEL_DIR, "custom_cnn.h5")
    model.save(out_path)
    print(f"\n✅ Custom CNN saved → {out_path}")
    print(f"   Training time: {elapsed/60:.1f} min")

    plot_history(hist.history, "Custom CNN — Training History",
                 os.path.join(PLOT_DIR, "custom_cnn_history.png"))

    # Save history as JSON for later analysis
    with open(os.path.join(MODEL_DIR, "custom_cnn_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in hist.history.items()}, f, indent=2)

    return model, hist.history


# ══════════════════════════════════════════════════════════════════════════════
# Train MobileNetV2 (Phase 1 → Phase 2)
# ══════════════════════════════════════════════════════════════════════════════
def train_mobilenetv2(train_gen, val_gen):
    print("\n" + "═" * 60)
    print("  Training  MobileNetV2  (transfer learning)")
    print("═" * 60)

    # ── Phase 1: frozen base, train head only ─────────────────────────────────
    print("\n── Phase 1: Head training (base frozen) ──")
    model = build_mobilenetv2(num_classes=len(CLASSES), img_size=IMG_SIZE, phase=1,
                               learning_rate=1e-3)

    callbacks_p1 = [
        EarlyStopping(monitor="val_accuracy", patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-5, verbose=1),
        CSVLogger(os.path.join(MODEL_DIR, "mobilenetv2_phase1_log.csv")),
    ]

    t0 = time.time()
    hist_p1 = model.fit(
        train_gen,
        epochs=MOB_EPOCHS_P1,
        validation_data=val_gen,
        callbacks=callbacks_p1,
        verbose=1,
    )
    print(f"   Phase 1 done  ({(time.time()-t0)/60:.1f} min)")

    # ── Phase 2: unfreeze top layers, fine-tune ───────────────────────────────
    print("\n── Phase 2: Fine-tuning top 54 layers ──")
    model = unfreeze_for_finetuning(model, n_layers=54, new_lr=1e-5)

    callbacks_p2 = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "mobilenetv2_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(MODEL_DIR, "mobilenetv2_phase2_log.csv")),
    ]

    hist_p2 = model.fit(
        train_gen,
        epochs=MOB_EPOCHS_P2,
        validation_data=val_gen,
        callbacks=callbacks_p2,
        verbose=1,
    )
    elapsed = time.time() - t0

    # Save final model
    out_path = os.path.join(MODEL_DIR, "mobilenetv2.h5")
    model.save(out_path)
    print(f"\n✅ MobileNetV2 saved → {out_path}")
    print(f"   Total training time: {elapsed/60:.1f} min")

    # Merge histories for plotting
    combined = merge_histories(hist_p1.history, hist_p2.history)
    plot_history(combined, "MobileNetV2 — Training History (Phase 1 + 2)",
                 os.path.join(PLOT_DIR, "mobilenetv2_history.png"))

    with open(os.path.join(MODEL_DIR, "mobilenetv2_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in combined.items()}, f, indent=2)

    return model, combined


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available     : {len(gpus)}")
    if gpus:
        for g in gpus:
            print(f"  {g}")
    else:
        print("  (running on CPU — training will be slow, ~2–4h)")

    # Load data
    train_gen, val_gen = get_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    # Train both models
    cnn_model,  cnn_hist  = train_custom_cnn(train_gen, val_gen)
    mob_model,  mob_hist  = train_mobilenetv2(train_gen, val_gen)

    # Final summary
    print("\n" + "═" * 60)
    print("  Training Complete — Summary")
    print("═" * 60)
    print(f"  Custom CNN   best val accuracy: "
          f"{max(cnn_hist['val_accuracy']):.4f}")
    print(f"  MobileNetV2  best val accuracy: "
          f"{max(mob_hist['val_accuracy']):.4f}")
    print(f"\n  Models saved to : {MODEL_DIR}")
    print(f"  Plots  saved to : {PLOT_DIR}")
    print("\n  Next step → python evaluate.py")
