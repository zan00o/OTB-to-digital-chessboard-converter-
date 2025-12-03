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
# logging & plotting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Define a small CNN model w/ tensorflow.keras
def build_small_cnn(num_classes: int, input_height: int = 192, input_width: int = 96):
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=(input_height, input_width, 3))
    x = inputs
    # 3 conv blocks with increasing filter features
    # each layer has Conv2D + BatchNorm + ReLU + MaxPool + Dropout of 15%
    for filters in [32, 64, 96]:
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
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

    # callbacks: reduce LR on plateau, early stopping, and catch NaN issues
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),  # stop if gradients explode
    ]

    # arguments: dataset path, output model path, image size, batch size, epochs
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=str,
                    help="folder with 13 class subfolders (created by build_dataset.py)")
    ap.add_argument("--out", required=True, type=str,
                    help="path to save Keras model, e.g., models/classifier.keras")
    ap.add_argument("--img-width", type=int, default=96)
    ap.add_argument("--img-height", type=int, default=192, help="Height of input images (default 192 for 2:1 ratio)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    image_size = (args.img_height, args.img_width)  # TensorFlow uses (height, width)

    # Class weighting
    dataset_path = pathlib.Path(args.dataset_root)

    class_counts = {}
    for class_dir in sorted(dataset_path.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.png")))
            class_counts[class_dir.name] = count
    
    total_samples = sum(class_counts.values())
    num_classes_count = len(class_counts)
    
    # Higher weight for rare classes (queens), lower for common (empty)
    class_weight = {}
    for idx, class_name in enumerate(sorted(class_counts.keys())):
        count = class_counts[class_name]
        # class weight inversely proportional to frequency in dataset, capped at 10.0
        raw_weight = total_samples / (num_classes_count * count)
        class_weight[idx] = min(10.0, raw_weight)
    
    print("Class counts:", class_counts)
    print("Class weights:", {sorted(class_counts.keys())[i]: f"{w:.2f}" for i, w in class_weight.items()})

    # create raw datasets from directory
    # 80% train, 20% val split
    train_raw = tf.keras.utils.image_dataset_from_directory(
        args.dataset_root, validation_split=0.2, subset="training",
        seed=67, image_size=image_size, batch_size=args.batch_size, # lol
        labels="inferred", label_mode="categorical", shuffle=True
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        args.dataset_root, validation_split=0.2, subset="validation",
        seed=67, image_size=image_size, batch_size=args.batch_size,
        labels="inferred", label_mode="categorical", shuffle=True
    )

    class_names = list(train_raw.class_names)
    num_classes = len(class_names)

    # Augmentation & normalization layers (aggressive to improve generalization)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),        # increased from 0.05
        tf.keras.layers.RandomZoom(0.15),           # increased from 0.1
        tf.keras.layers.RandomContrast(0.2),        # increased from 0.1
        tf.keras.layers.RandomBrightness(0.2),      # NEW: handle lighting variation
    ])
    def normalize(x): return tf.cast(x, tf.float32) / 255.0

    # Prepare datasets
    AUTOTUNE = tf.data.AUTOTUNE

    # Create augmented and normalized datasets
    train_ds = (train_raw
                .map(lambda x, y: (aug(normalize(x)), y), num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds = (val_raw
              .map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)
              .prefetch(AUTOTUNE))

    # Build & train model with class weights
    model = build_small_cnn(num_classes=num_classes, input_height=args.img_height, input_width=args.img_width)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight, callbacks=callbacks)

    # Save model
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))

    with open(str(out_path.with_suffix(".classes.json")), "w") as f:
        json.dump(class_names, f, indent=2)

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(out_path.with_suffix('.curves.png')))
    print(f"Saved training curves to {out_path.with_suffix('.curves.png')}")

    # confusion matrix on validation set
    y_true, y_pred = [], []
    for images, labels in val_raw:
        preds = model.predict(tf.cast(images, tf.float32) / 255.0, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    
    # Plot 
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(str(out_path.with_suffix('.confusion.png')))
    print(f"Saved confusion matrix to {out_path.with_suffix('.confusion.png')}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print(f"Saved model to {out_path}")
    print("Classes:", class_names)

if __name__ == "__main__":
    main()
