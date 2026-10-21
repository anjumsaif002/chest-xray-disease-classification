import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from src.dataset import CLASS_NAMES, LABEL_TO_INDEX, INDEX_TO_LABEL

CANCER_CLASSES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "squamous_cell_carcinoma"
]

def predict_single_image(image_path, model_path, img_size=(224, 224), show_plot=False):
    """
    Runs model inference on an image and outputs detailed disease probabilities.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = keras.models.load_model(model_path)

    # Load and preprocess
    image = keras.utils.load_img(image_path, target_size=img_size)
    image_array = keras.utils.img_to_array(image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Predict
    probabilities = model.predict(image_batch)[0]
    predicted_idx = np.argmax(probabilities)
    predicted_class = INDEX_TO_LABEL[predicted_idx]
    confidence = probabilities[predicted_idx] * 100

    # Overall cancer vs normal assessment
    cancer_prob = sum(probabilities[LABEL_TO_INDEX[c]] for c in CANCER_CLASSES) * 100
    normal_prob = probabilities[LABEL_TO_INDEX["normal"]] * 100

    if predicted_class == "normal":
        verdict = "Negative for Pulmonary Malignancy (Normal)"
    else:
        verdict = "Pulmonary Lesion / Malignancy Detected"

    if confidence >= 80:
        conf_level = "High Confidence"
    elif confidence >= 60:
        conf_level = "Medium Confidence"
    else:
        conf_level = "Low Confidence (Review Suggested)"

    print("=" * 60)
    print("           CHEST SCAN ANALYSIS RESULT")
    print("=" * 60)
    print(f"File:                  {os.path.basename(image_path)}")
    print(f"Primary Diagnosis:     {predicted_class.replace('_', ' ').title()}")
    print(f"Diagnostic Confidence: {confidence:.2f}% ({conf_level})")
    print(f"Clinical Summary:      {verdict}")
    print("-" * 60)
    print(f"Combined Cancer Risk:  {cancer_prob:.2f}%")
    print(f"Normal Tissue Prob:    {normal_prob:.2f}%")
    print("-" * 60)
    print("Class Breakdown:")
    for i, prob in enumerate(probabilities):
        name = INDEX_TO_LABEL[i].replace("_", " ").title()
        bar = "#" * int(prob * 30)
        print(f"  - {name:<25}: {prob * 100:6.2f}% | {bar}")
    print("=" * 60)

    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"{verdict}\nType: {predicted_class} ({confidence:.1f}%)")
        plt.tight_layout()
        plt.show()

    return predicted_class, confidence, probabilities

def main():
    parser = argparse.ArgumentParser(description="Predict Disease from Chest CT / X-Ray Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input scan image")
    parser.add_argument("--model", type=str, default="best_model.keras", help="Path to trained .keras model")
    parser.add_argument("--show", action="store_true", help="Display image with prediction overlay")
    args = parser.parse_args()

    predict_single_image(args.image, args.model, show_plot=args.show)

if __name__ == "__main__":
    main()
