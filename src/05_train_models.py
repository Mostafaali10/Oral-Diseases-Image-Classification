"""
Phase 7: Model Training (GPU - RTX)
- Custom CNN
- MobileNetV2
- EfficientNetB0
- ResNet50
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from sklearn.utils.class_weight import compute_class_weight

# ====================== GPU SETUP ======================
print("=" * 60)
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  → GPU {i}: {gpu.name}")
        # تأكيد إن TensorFlow هيستخدم الـ GPU
        print("Running on GPU (RTX)")
    except RuntimeError as e:
        print("GPU setup error:", e)
else:
    print("WARNING: No GPU detected. Training will run on CPU (very slow).")
print("=" * 60)

# ====================== CONFIG ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32          # ممكن تزوده لـ 64 لو الـ VRAM تسمح
EPOCHS = 40
SEED = 42
# ====================================================

tf.keras.utils.set_random_seed(SEED)

def make_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        SPLITS_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        seed=SEED
    )
    val_ds = keras.utils.image_dataset_from_directory(
        SPLITS_DIR / "val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        seed=SEED
    )
    test_ds = keras.utils.image_dataset_from_directory(
        SPLITS_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def get_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_custom_cnn(num_classes):
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = get_augmentation()(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="CustomCNN")


def build_transfer_model(base_fn, num_classes, name):
    base = base_fn(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = get_augmentation()(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name=name)


def train_model(model, train_ds, val_ds, class_weight, model_name):
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    ckpt = callbacks.ModelCheckpoint(
        MODEL_DIR / f"{model_name}_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    es = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    reduce = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[ckpt, es, reduce],
        verbose=1
    )
    return history


def main():
    train_ds, val_ds, test_ds, class_names = make_datasets()
    num_classes = len(class_names)

    # Class weights
    y_train = np.concatenate([y.numpy() for _, y in train_ds])
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(weights))
    print("Class weights:", class_weight)

    # -------------------- 1. Custom CNN --------------------
    print("\n" + "=" * 60)
    print("Training Custom CNN on GPU")
    print("=" * 60)
    cnn = build_custom_cnn(num_classes)
    train_model(cnn, train_ds, val_ds, class_weight, "CustomCNN")

    # -------------------- 2. MobileNetV2 --------------------
    print("\n" + "=" * 60)
    print("Training MobileNetV2 on GPU")
    print("=" * 60)
    mobilenet = build_transfer_model(MobileNetV2, num_classes, "MobileNetV2")
    train_model(mobilenet, train_ds, val_ds, class_weight, "MobileNetV2")

    # -------------------- 3. EfficientNetB0 --------------------
    print("\n" + "=" * 60)
    print("Training EfficientNetB0 on GPU")
    print("=" * 60)
    eff = build_transfer_model(EfficientNetB0, num_classes, "EfficientNetB0")
    train_model(eff, train_ds, val_ds, class_weight, "EfficientNetB0")

    # -------------------- 4. ResNet50 --------------------
    print("\n" + "=" * 60)
    print("Training ResNet50 on GPU")
    print("=" * 60)
    resnet = build_transfer_model(ResNet50, num_classes, "ResNet50")
    train_model(resnet, train_ds, val_ds, class_weight, "ResNet50")

    print("\n" + "=" * 60)
    print("All models trained successfully!")
    print("Models saved in:", MODEL_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()