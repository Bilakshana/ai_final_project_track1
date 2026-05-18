"""
evaluate.py — Comprehensive evaluation & error analysis for both models.

Run:
    python evaluate.py

Outputs:
    plots/confusion_matrix_custom_cnn.png
    plots/confusion_matrix_mobilenetv2.png
    plots/comparison_table.png
    plots/error_analysis.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from preprocess import get_generators, CLASSES

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "data/"
MODEL_DIR = "models/"
PLOT_DIR  = "plots/"
IMG_SIZE  = 224
BATCH_SIZE = 32

os.makedirs(PLOT_DIR, exist_ok=True)

# Known confusion patterns (for error analysis narrative)
CONFUSION_PAIRS = {
    ("glass",  "metal"):     "Both materials share a shiny, reflective surface. "
                             "The CNN cannot distinguish by material — only texture/colour.",
    ("paper",  "cardboard"): "Similar beige/brown tones and flat surfaces. "
                             "Cardboard is thicker but that's invisible in 2-D images.",
    ("plastic","glass"):     "Transparent plastic bags and clear glass bottles "
                             "look nearly identical under certain lighting.",
    ("trash",  "paper"):     "'Trash' is a catch-all. Crumpled paper and fast-food "
                             "wrappers appear in both categories.",
}


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, val_ds, model_name: str):
    """
    Run model on the full validation set, return per-class metrics dict.
    """
    print(f"\n── Evaluating: {model_name} ──")

    y_pred_probs = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Extract true labels from the dataset
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in val_ds])

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                 target_names=CLASSES, zero_division=0))

    return {
        "name":       model_name,
        "y_true":     y_true,
        "y_pred":     y_pred,
        "y_probs":    y_pred_probs,
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "cm":         confusion_matrix(y_true, y_pred),
        "filepaths":  [],   # not available with tf.data
    }


# ══════════════════════════════════════════════════════════════════════════════
# Confusion matrix plot
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)

    # Print cell values; white text on dark cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Confusion matrix → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Side-by-side comparison table
# ══════════════════════════════════════════════════════════════════════════════
def plot_comparison_table(results_list: list, save_path: str):
    """Bar chart comparing Accuracy, Precision, Recall, F1 across models."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n = len(results_list)
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, res in enumerate(results_list):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width - (n - 1) * width / 2,
                      vals, width,
                      label=res["name"],
                      color=colours[i % len(colours)],
                      alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="90% line")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Comparison table → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Error analysis
# ══════════════════════════════════════════════════════════════════════════════
def error_analysis(results: dict, save_path: str, top_n: int = 5):
    """
    Find the most-confused class pairs and plot a summary bar chart.
    (Image previews not available with tf.data pipelines — uses chart instead.)
    """
    from collections import Counter

    y_true     = results["y_true"]
    y_pred     = results["y_pred"]
    model_name = results["name"]

    wrong_mask = y_true != y_pred
    wrong_true = y_true[wrong_mask]
    wrong_pred = y_pred[wrong_mask]

    pair_counts = Counter(zip(wrong_true, wrong_pred))
    top_pairs   = pair_counts.most_common(top_n)

    print(f"\n── Error Analysis: {model_name} ──")
    print(f"  Total misclassified: {wrong_mask.sum()} / {len(y_true)}")
    print("  Top confusion pairs:")
    for (t, p), count in top_pairs:
        print(f"    True={CLASSES[t]:10s}  Pred={CLASSES[p]:10s}  ({count} times)")

    if not top_pairs:
        print("  No errors found.")
        return

    # ── Bar chart of confusion pairs ──────────────────────────────────────────
    labels = [f"{CLASSES[t]} →\n{CLASSES[p]}" for (t, p), _ in top_pairs]
    counts = [c for _, c in top_pairs]
    colours = ["#EF5350", "#FF7043", "#FFA726", "#FFCA28", "#66BB6A"][:len(top_pairs)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], counts[::-1], color=colours[::-1], edgecolor="white")

    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Number of misclassifications", fontsize=11)
    ax.set_title(f"Top Confusion Pairs — {model_name}\n"
                 f"({wrong_mask.sum()} total errors out of {len(y_true)} samples)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(counts) * 1.18)

    # Add explanation annotations
    for i, ((true_cls, pred_cls), _) in enumerate(reversed(top_pairs)):
        key = (CLASSES[true_cls], CLASSES[pred_cls])
        rev = (CLASSES[pred_cls], CLASSES[true_cls])
        tip = CONFUSION_PAIRS.get(key, CONFUSION_PAIRS.get(rev, "Visual similarity."))
        short_tip = tip[:60] + "…" if len(tip) > 60 else tip
        ax.text(0.5, i, f"  💡 {short_tip}",
                va="center", fontsize=7, color="#555", style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Error analysis → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _, val_gen = get_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    all_results = []
    for model_name, model_file in [
        ("Custom CNN",   "custom_cnn.h5"),
        ("MobileNetV2",  "mobilenetv2.h5"),
    ]:
        path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(path):
            print(f"⚠️  {path} not found — skipping. Run train.py first.")
            continue

        model = tf.keras.models.load_model(path)
        res   = evaluate_model(model, val_gen, model_name)
        all_results.append(res)

        plot_confusion_matrix(
            res["cm"],
            f"Confusion Matrix — {model_name}",
            os.path.join(PLOT_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
        )
        error_analysis(
            res,
            os.path.join(PLOT_DIR, f"error_analysis_{model_name.lower().replace(' ', '_')}.png")
        )

    if len(all_results) >= 2:
        plot_comparison_table(all_results,
                              os.path.join(PLOT_DIR, "comparison_table.png"))

    # Print final text summary
    print("\n" + "═" * 55)
    print("  Final Results Summary")
    print("═" * 55)
    print(f"  {'Model':<18} {'Accuracy':>9} {'F1':>9} {'Precision':>10} {'Recall':>8}")
    print("  " + "─" * 53)
    for r in all_results:
        print(f"  {r['name']:<18} {r['accuracy']:>9.4f} {r['f1']:>9.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f}")
    print()