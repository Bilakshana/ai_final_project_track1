"""
models.py — Model definitions for Track 1: Waste Classification

Two architectures:
  1. build_custom_cnn()   — Trained from scratch (baseline)
  2. build_mobilenetv2()  — Transfer learning from ImageNet (fine-tuned)

Usage:
    from models import build_custom_cnn, build_mobilenetv2

    cnn   = build_custom_cnn(num_classes=6, img_size=224)
    mobv2 = build_mobilenetv2(num_classes=6, img_size=224)
"""

import tensorflow as tf
from keras import layers, models, regularizers
from keras.applications import MobileNetV2
from keras.optimizers import Adam


NUM_CLASSES = 6
IMG_SIZE    = 224


# ══════════════════════════════════════════════════════════════════════════════
# Model 1 — Custom CNN (trained from scratch)
# ══════════════════════════════════════════════════════════════════════════════

def build_custom_cnn(num_classes: int = NUM_CLASSES,
                     img_size: int = IMG_SIZE,
                     learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    3-block CNN baseline.

    Each block:  Conv2D(ReLU) → BatchNorm → MaxPool → Dropout
    Classifier:  GlobalAvgPool → Dense(256) → Dense(num_classes, softmax)

    Why GlobalAvgPool instead of Flatten?
      • Fewer parameters → less overfitting on ~2500 training images
      • Spatially invariant — good for waste objects photographed at
        different angles / distances
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")
    x = inputs

    # ── Block 1 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.30, name="drop2")(x)

    # ── Block 3 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.35, name="drop3")(x)

    # ── Classifier head ───────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="fc1")(x)
    x = layers.Dropout(0.50, name="drop_fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="Custom_CNN")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Model 2 — MobileNetV2  (transfer learning + fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════

def build_mobilenetv2(num_classes: int = NUM_CLASSES,
                      img_size: int = IMG_SIZE,
                      phase: int = 1,
                      learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    MobileNetV2 transfer-learning model.

    Phase 1 (phase=1):
      Base frozen → only train the new classification head.
      Use a normal LR (1e-3). Fast convergence in ~10 epochs.

    Phase 2 (phase=2):
      Top 54 layers unfrozen → fine-tune with tiny LR (1e-5).
      Call unfreeze_top_layers() after Phase 1 training, then
      re-compile and continue training.

    Args:
        num_classes   : number of output classes
        img_size      : spatial dimension (224 recommended)
        phase         : 1 = frozen base, 2 = partial unfreeze
        learning_rate : optimizer LR (use 1e-3 for P1, 1e-5 for P2)
    """
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,          # remove ImageNet classifier head
        weights="imagenet",
    )
    base_model.trainable = False    # Phase 1: freeze all base layers

    if phase == 2:
        # Unfreeze top 54 layers for fine-tuning
        for layer in base_model.layers[-54:]:
            layer.trainable = not isinstance(layer, layers.BatchNormalization)
            # Keep BN frozen — running stats from ImageNet are valuable

    # ── New classification head ───────────────────────────────────────────────
    inputs  = layers.Input(shape=(img_size, img_size, 3), name="input_image")
    x       = base_model(inputs, training=(phase == 2))
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dense(256, activation="relu", name="fc1")(x)
    x       = layers.Dropout(0.40, name="drop_fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"MobileNetV2_phase{phase}")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_for_finetuning(model: tf.keras.Model,
                             n_layers: int = 54,
                             new_lr: float = 1e-5) -> tf.keras.Model:
    """
    Unfreeze the top `n_layers` of the MobileNetV2 base inside an existing
    model and re-compile with a tiny learning rate for Phase 2 fine-tuning.

    Call this after Phase 1 training completes.
    """
    # Find the MobileNetV2 base layer (first Functional sub-layer)
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            base = layer
            break

    if base is None:
        raise ValueError("Could not find MobileNetV2 base inside the model.")

    base.trainable = True
    # Freeze everything except the last n_layers
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    # Keep all BatchNorm layers frozen — do not update running stats
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=new_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"✅ Unfrozen top {n_layers} layers. LR = {new_lr}")
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"   Trainable parameters: {trainable:,}")
    return model


# ── CLI sanity check ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Custom CNN ──")
    cnn = build_custom_cnn()
    cnn.summary()

    print("\n── MobileNetV2 Phase 1 ──")
    mob = build_mobilenetv2(phase=1)
    mob.summary()
