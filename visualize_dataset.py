"""
visualize_dataset.py
--------------------
Generates a grid image showing random sample leaves from each class
in the PlantVillage dataset. Useful for visual inspection before
training and makes a nice figure for the README / report.

Usage:
    python visualize_dataset.py

Output: results/dataset_samples.png

Authors: Ajinkya Awari, Akash Raskar, Shrirameshwar Patil, Namrata Jamdar
SKNCOE, Pune | 2022-2023
"""

import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from config import DATASET_PATH, RESULTS_DIR, CLASSES_TO_USE, SEED

random.seed(SEED)


def get_classes():
    """Return the list of class folder names to display."""
    if CLASSES_TO_USE:
        return [c for c in CLASSES_TO_USE
                if os.path.isdir(os.path.join(DATASET_PATH, c))]
    # fall back to whatever folders exist in the dataset dir
    return sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])


def pick_samples(class_dir, n=3):
    """Return paths to n random images from a class folder."""
    files = [f for f in os.listdir(class_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if len(files) == 0:
        return []
    chosen = random.sample(files, min(n, len(files)))
    return [os.path.join(class_dir, f) for f in chosen]


def main():
    if not os.path.isdir(DATASET_PATH):
        print(f"[ERROR] Dataset folder not found at: {DATASET_PATH}")
        print("Download PlantVillage from Kaggle first.")
        return

    classes = get_classes()
    if not classes:
        print("[ERROR] No class folders found.")
        return

    samples_per_class = 3
    rows = len(classes)
    cols = samples_per_class

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.8))
    if rows == 1:
        axes = [axes]

    for row_idx, cls_name in enumerate(classes):
        cls_path = os.path.join(DATASET_PATH, cls_name)
        samples = pick_samples(cls_path, samples_per_class)
        nice_name = cls_name.replace("___", "\n").replace("_", " ")

        for col_idx in range(cols):
            ax = axes[row_idx][col_idx] if cols > 1 else axes[row_idx]
            if col_idx < len(samples):
                img = Image.open(samples[col_idx]).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                        fontsize=10, color='grey',
                        transform=ax.transAxes)
            ax.axis('off')
            # label only the first column
            if col_idx == 0:
                ax.set_title(nice_name, fontsize=8, fontweight='bold',
                             loc='left', pad=4)

    fig.suptitle("PlantVillage Dataset - Sample Images per Class",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "dataset_samples.png")
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  ({rows} classes x {cols} samples)")


if __name__ == "__main__":
    main()
