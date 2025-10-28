"""
train_classifier.py
Train and save a CNN classifier for chess piece/square images.
Assumes dataset is organized in subfolders per class, e.g.:
dataset-root/
    empty/
    color_piece 
Hyperparameters:
- image size: 96x96
- batch size: 64
- epochs: 12
- Dropout: 0.15 after conv layers, 0.25 before dense
- Augmentations: random flip, rotation, zoom, contrast
- ReLU activations, batch normalization
- Adam optimizer, categorical crossentropy loss
- Softmax output for multi-class classification
"""

import argparse, pathlib, json, tensorflow as tf

# Define a small CNN model w/ tensorflow.keras
def build_small_cnn(num_classes: int, input_size: int = 96):
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=(input_size, input_size, 3))
    x = inputs
    # 3 conv blocks with increasing filter features
    # each layer has Conv2D + BatchNorm + ReLU + MaxPool + Dropout of 15%
    for filters in [32, 64, 96]:
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15)(x)
    # Final layers
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    # arguments: dataset path, output model path, image size, batch size, epochs
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=str,
                    help="folder with 13 class subfolders (created by build_dataset.py)")
    ap.add_argument("--out", required=True, type=str,
                    help="path to save Keras model, e.g., models/classifier.keras")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    image_size = (args.img_size, args.img_size)

    # create raw datasets from directory
    # 80% train, 20% val split
    train_raw = tf.keras.utils.image_dataset_from_directory(
        args.dataset_root, validation_split=0.2, subset="training",
        seed=1337, image_size=image_size, batch_size=args.batch_size,
        labels="inferred", label_mode="categorical", shuffle=True
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        args.dataset_root, validation_split=0.2, subset="validation",
        seed=1337, image_size=image_size, batch_size=args.batch_size,
        labels="inferred", label_mode="categorical", shuffle=True
    )

    class_names = list(train_raw.class_names)
    num_classes = len(class_names)

    # Augmentation & normalization layers
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    def normalize(x): return tf.cast(x, tf.float32) / 255.0

    # Prepare datasets
    # AUTOTUNE allows the dataset to fetch batches in the background while the model is training
    AUTOTUNE = tf.data.AUTOTUNE

    # Create augmented and normalized datasets
    train_ds = (train_raw
                .map(lambda x, y: (aug(normalize(x)), y), num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds = (val_raw
              .map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)
              .prefetch(AUTOTUNE))

    # Build & train model with the created datasets
    model = build_small_cnn(num_classes=num_classes, input_size=args.img_size)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))

    with open(str(out_path.with_suffix(".classes.json")), "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved model to {out_path}")
    print("Classes:", class_names)

if __name__ == "__main__":
    main()
