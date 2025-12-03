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
from sklearn.model_selection import train_test_split
import shutil

# Define a small CNN model w/ tensorflow.keras
def build_transfer_model(num_classes: int, input_size: int = 96):
    from tensorflow.keras import layers, models
    
    # Load pretrained MobileNetV2 (more stable than EfficientNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    inputs = layers.Input(shape=(input_size, input_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():

    # callbacks: reduce LR on plateau, early stopping, and catch NaN issues
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),  # stop if gradients explode
    ]

    # arguments: dataset path, output model path, image size, batch size, epochs
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=str,
                    help="folder with 13 class subfolders (created by build_dataset.py)")
    ap.add_argument("--out", required=True, type=str,
                    help="path to save Keras model, e.g., models/classifier.keras")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=64) # on gpu 256 is fine, on cpu 64 is fine
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    image_size = (args.img_size, args.img_size)

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

    # Create STRATIFIED train/val split by manually splitting files per class
    # This ensures each class is represented in both train and val sets
    print("\nCreating stratified train/val split...")
    train_dir = pathlib.Path("data/dataset/train_split")
    val_dir = pathlib.Path("data/dataset/val_split")
    
    # Clean up existing split dirs
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # For each class, split files 80/20 and create symlinks
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        files = list(class_dir.glob("*.png"))
        
        # Stratified split
        train_files, val_files = train_test_split(
            files, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Create class directories
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        
        # Create symlinks (or copy on Windows if symlinks fail)
        for f in train_files:
            dst = train_dir / class_name / f.name
            try:
                dst.symlink_to(f.resolve())
            except OSError:
                shutil.copy2(f, dst)
        
        for f in val_files:
            dst = val_dir / class_name / f.name
            try:
                dst.symlink_to(f.resolve())
            except OSError:
                shutil.copy2(f, dst)
        
        print(f"  {class_name:20s}: {len(train_files):4d} train, {len(val_files):4d} val")
    
    # Load datasets from split directories
    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=image_size, batch_size=args.batch_size,
        labels="inferred", label_mode="categorical", shuffle=True, seed=42
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=image_size, batch_size=args.batch_size,
        labels="inferred", label_mode="categorical", shuffle=True, seed=42
    )

    class_names = list(train_raw.class_names)
    num_classes = len(class_names)

    # NO augmentation - testing with identity layer
    aug = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x)  # Identity - does nothing
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
    model = build_transfer_model(num_classes=num_classes, input_size=args.img_size)
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
