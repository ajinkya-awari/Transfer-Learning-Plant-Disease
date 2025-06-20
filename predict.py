"""
predict.py
----------
Load a trained model and classify a single leaf image from the command line.

Usage:
    python predict.py <image_path> [model_name]

Examples:
    python predict.py my_leaf.jpg
    python predict.py samples/tomato.jpg ResNet50

Available models: VGG16 | ResNet50 | EfficientNetB0  (default: EfficientNetB0)

Author:  Ajinkya Avinash Awari
Guide:   Prof. Vrushali Paithankar  |  SKNCOE, Pune  |  SKNCOE, Pune
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from config import MODELS_DIR, RESULTS_DIR, IMAGE_SIZE


def load_class_names():
    """Read the class-index mapping saved during training."""
    path = os.path.join(RESULTS_DIR, "class_names.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "class_names.json not found in results/. "
            "Run train_comparison.py first to generate it."
        )
    with open(path) as f:
        mapping = json.load(f)
    return {v: k for k, v in mapping.items()}


def predict_image(image_path, model_name="EfficientNetB0"):
    """Load a saved .h5 model and print the predicted disease."""
    h5 = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    if not os.path.exists(h5):
        print(f"[ERROR] Model not found: {h5}")
        print(f"Files in '{MODELS_DIR}/':")
        for f in os.listdir(MODELS_DIR):
            print(f"  - {f}")
        return

    print(f"\nLoading model: {model_name}")
    model = tf.keras.models.load_model(h5)

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
    inp = np.expand_dims(arr, axis=0)

    preds    = model.predict(inp, verbose=0)[0]
    top_idx  = int(np.argmax(preds))
    conf     = float(preds[top_idx]) * 100
    cls_map  = load_class_names()
    disease  = cls_map.get(top_idx, f"Class {top_idx}")
    readable = disease.replace("___", " - ").replace("_", " ")

    print(f"\n{'=' * 50}")
    print(f"  IMAGE      : {os.path.basename(image_path)}")
    print(f"  MODEL      : {model_name}")
    print(f"  PREDICTION : {readable}")
    print(f"  CONFIDENCE : {conf:.1f}%")
    print(f"{'=' * 50}\n")

    # top 3
    top3 = np.argsort(preds)[::-1][:3]
    print("  Top 3 predictions:")
    for rank, idx in enumerate(top3, 1):
        name = cls_map.get(int(idx), f"Class {idx}")
        name = name.replace("___", " - ").replace("_", " ")
        print(f"    {rank}. {name:<40s} {preds[idx]*100:.1f}%")

    # quick visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(img)
    ax1.set_title("Input Image", fontsize=13)
    ax1.axis('off')

    top3_names = [
        cls_map.get(int(i), f"Class {i}").replace("___", "\n").replace("_", " ")
        for i in top3
    ]
    top3_vals = [preds[i] * 100 for i in top3]
    bar_cols = ['#2196F3' if j == 0 else '#90CAF9' for j in range(3)]

    bars = ax2.barh(top3_names[::-1], top3_vals[::-1], color=bar_cols[::-1])
    ax2.set_xlabel("Confidence (%)", fontsize=11)
    ax2.set_title(f"Prediction -- {model_name}", fontsize=13)
    ax2.set_xlim(0, 110)
    for bar, score in zip(bars, top3_vals[::-1]):
        ax2.text(score + 1, bar.get_y() + bar.get_height()/2,
                 f"{score:.1f}%", va='center', fontsize=10, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle("Plant Disease Detection -- SKNCOE Pune | IJARSCT 2023",
                 fontsize=12, fontstyle='italic')
    plt.tight_layout()
    out = "prediction_result.png"
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"  Result saved as: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python predict.py <image_path> [model_name]")
        print("\nExample:")
        print("  python predict.py my_leaf.jpg EfficientNetB0")
        print("\nAvailable models: VGG16 | ResNet50 | EfficientNetB0")
        sys.exit(1)

    img_path = sys.argv[1]
    mname    = sys.argv[2] if len(sys.argv) > 2 else "EfficientNetB0"
    predict_image(img_path, mname)
