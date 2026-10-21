import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma"
]

LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {i: name for name, i in LABEL_TO_INDEX.items()}

def normalize_label(folder_name):
    """Normalize dataset directory name to standard class label."""
    name = folder_name.lower()
    if "adenocarcinoma" in name:
        return "adenocarcinoma"
    elif "large.cell" in name or "large_cell" in name or "large cell" in name:
        return "large_cell_carcinoma"
    elif "normal" in name:
        return "normal"
    elif "squamous" in name:
        return "squamous_cell_carcinoma"
    return "unknown"

def create_dataframe_from_dir(split_dir):
    """Recursively parses image files and builds a DataFrame."""
    image_paths = []
    labels = []
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    if not os.path.exists(split_dir):
        return pd.DataFrame(columns=["path", "label", "label_index"])

    for folder in os.listdir(split_dir):
        folder_path = os.path.join(split_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        label = normalize_label(folder)
        if label == "unknown":
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_exts):
                image_paths.append(os.path.join(folder_path, filename))
                labels.append(label)

    df = pd.DataFrame({"path": image_paths, "label": labels})
    df["label_index"] = df["label"].map(LABEL_TO_INDEX)
    return df.dropna().reset_index(drop=True)

def load_image(image_path, label, img_size=(224, 224)):
    """Loads and normalizes an image file for training."""
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_tf_dataset(df, img_size=(224, 224), batch_size=32, shuffle=False):
    """Converts a pandas DataFrame into an optimized tf.data.Dataset."""
    paths = df["path"].values
    labels = df["label_index"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=42)

    dataset = dataset.map(
        lambda p, l: load_image(p, l, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_class_weights(df):
    """Computes balanced class weights to mitigate class imbalance."""
    classes = np.unique(df["label_index"])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=df["label_index"]
    )
    return {c: float(w) for c, w in zip(classes, weights)}
