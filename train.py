import os
import argparse
import matplotlib.pyplot as plt
from tensorflow import keras

from src.dataset import (
    CLASS_NAMES,
    create_dataframe_from_dir,
    create_tf_dataset,
    get_class_weights
)
from src.model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Chest Disease Classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder containing train/valid/test")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_model", type=str, default="best_model.keras", help="Path to save trained model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    test_dir = os.path.join(args.data_dir, "test")

    print(f"[+] Loading datasets from {args.data_dir}...")
    train_df = create_dataframe_from_dir(train_dir)
    valid_df = create_dataframe_from_dir(valid_dir)
    test_df = create_dataframe_from_dir(test_dir)

    if len(train_df) == 0:
        print(f"[-] Error: No training images found in {train_dir}!")
        print("    Please ensure data folder has train/, valid/, and test/ subdirectories.")
        return

    print(f"    Train images: {len(train_df)}")
    print(f"    Validation images: {len(valid_df)}")
    print(f"    Test images: {len(test_df)}")

    class_weights = get_class_weights(train_df)
    print(f"[+] Calculated class weights: {class_weights}")

    train_ds = create_tf_dataset(train_df, batch_size=args.batch_size, shuffle=True)
    val_ds = create_tf_dataset(valid_df, batch_size=args.batch_size, shuffle=False)
    test_ds = create_tf_dataset(test_df, batch_size=args.batch_size, shuffle=False)

    print("[+] Building Transfer Learning MobileNetV2 Model...")
    model = build_model(num_classes=len(CLASS_NAMES), learning_rate=args.lr)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.output_model,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"[+] Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("[+] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"[+] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print(f"[+] Best model checkpoint saved to: {args.output_model}")

if __name__ == "__main__":
    main()
