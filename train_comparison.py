"""
train_comparison.py
-------------------
Trains and compares VGG16, ResNet50, and EfficientNetB0 on the PlantVillage
dataset using transfer learning. Generates comparison charts and a CSV
summary with accuracy, precision, recall, and F1 scores.

Part of the Plant Disease Detection project -- SKNCOE, Pune (2022-2023).
Published: IJARSCT Vol.3, Issue 2 & 4, April 2023
DOI: 10.48175/IJARSCT-9156

Author:  Ajinkya Avinash Awari
Guide:   Prof. Vrushali Paithankar  |  SKNCOE, Pune
"""

import os
import csv
import json
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from config import (
    DATASET_PATH, RESULTS_DIR, MODELS_DIR,
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, VAL_SPLIT, TEST_SPLIT,
    LEARNING_RATE, SEED, CLASSES_TO_USE,
)

warnings.filterwarnings('ignore')
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ────────────────────────────────────────────────────────────────
#  Data loading
# ────────────────────────────────────────────────────────────────
def prepare_data():
    """Read images from disk and return train / val generators."""
    print("\n" + "=" * 60)
    print("  LOADING DATASET")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset folder '{DATASET_PATH}' not found. "
            "Download PlantVillage from Kaggle and place the class folders "
            "inside a 'dataset/' directory next to this script."
        )

    if CLASSES_TO_USE:
        missing = [c for c in CLASSES_TO_USE
                   if not os.path.isdir(os.path.join(DATASET_PATH, c))]
        if missing:
            print("[WARNING] These classes were not found:")
            for m in missing:
                print(f"  - {m}")
        available = [c for c in CLASSES_TO_USE
                     if os.path.isdir(os.path.join(DATASET_PATH, c))]
        if not available:
            raise FileNotFoundError("No valid class folders found!")

    hold_out = VAL_SPLIT + TEST_SPLIT

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=hold_out,
    )
    val_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=hold_out,
    )

    classes = CLASSES_TO_USE if CLASSES_TO_USE else None

    train_gen = train_aug.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        classes=classes,
        seed=SEED,
        shuffle=True,
    )
    val_gen = val_aug.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=classes,
        seed=SEED,
        shuffle=False,
    )

    num_classes = len(train_gen.class_indices)
    print(f"\n  Classes found   : {num_classes}")
    print(f"  Training images : {train_gen.samples}")
    print(f"  Validation imgs : {val_gen.samples}")
    print(f"  Class list: {list(train_gen.class_indices.keys())}")

    with open(os.path.join(RESULTS_DIR, "class_names.json"), "w") as fp:
        json.dump(train_gen.class_indices, fp, indent=2)

    return train_gen, val_gen, num_classes


# ────────────────────────────────────────────────────────────────
#  Model builders
# ────────────────────────────────────────────────────────────────
def _attach_head(base, num_classes, name):
    """Freeze a pretrained base and add our own classification head."""
    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ], name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_vgg16(n):
    base = VGG16(weights='imagenet', include_top=False,
                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.trainable = False
    return _attach_head(base, n, "VGG16_PlantDisease")


def build_resnet50(n):
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.trainable = False
    return _attach_head(base, n, "ResNet50_PlantDisease")


def build_efficientnet(n):
    base = EfficientNetB0(weights='imagenet', include_top=False,
                          input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.trainable = False
    return _attach_head(base, n, "EfficientNetB0_PlantDisease")


# ────────────────────────────────────────────────────────────────
#  Training
# ────────────────────────────────────────────────────────────────
def train_model(model, train_gen, val_gen, model_name):
    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {model_name}")
    print(f"{'=' * 60}")
    model.summary()

    cbs = [
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{model_name}_best.h5"),
            monitor='val_accuracy', save_best_only=True, verbose=1,
        ),
    ]

    t0 = time.time()
    history = model.fit(train_gen, epochs=EPOCHS,
                        validation_data=val_gen, callbacks=cbs, verbose=1)
    elapsed = time.time() - t0
    print(f"\n  {model_name} finished in {elapsed/60:.1f} min")
    return history, elapsed


# ────────────────────────────────────────────────────────────────
#  Evaluation
# ────────────────────────────────────────────────────────────────
def evaluate_model(model, val_gen, model_name):
    print(f"\n  Evaluating {model_name} ...")
    val_gen.reset()
    y_prob = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = val_gen.classes

    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    report = classification_report(
        y_true, y_pred,
        target_names=list(val_gen.class_indices.keys()),
        output_dict=True, zero_division=0,
    )
    acc  = report['accuracy']
    prec = report['weighted avg']['precision']
    rec  = report['weighted avg']['recall']
    f1   = report['weighted avg']['f1-score']

    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")

    # save per-class report
    txt = classification_report(
        y_true, y_pred,
        target_names=list(val_gen.class_indices.keys()),
        zero_division=0,
    )
    with open(os.path.join(RESULTS_DIR, f"{model_name}_report.txt"), "w") as fp:
        fp.write(f"Classification Report -- {model_name}\n")
        fp.write("=" * 60 + "\n")
        fp.write(txt)

    return acc, prec, rec, f1, y_true, y_pred


# ────────────────────────────────────────────────────────────────
#  Plotting helpers
# ────────────────────────────────────────────────────────────────
PALETTE = ['#2196F3', '#4CAF50', '#FF5722']


def plot_training_history(histories, names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (h, name) in enumerate(zip(histories, names)):
        acc, vacc = h.history['accuracy'], h.history['val_accuracy']
        loss, vloss = h.history['loss'], h.history['val_loss']
        ep = range(1, len(acc) + 1)

        axes[0].plot(ep, vacc, color=PALETTE[i], lw=2.5, label=f'{name} (val)')
        axes[0].plot(ep, acc, color=PALETTE[i], lw=1.5, ls='--', alpha=0.6,
                     label=f'{name} (train)')
        axes[1].plot(ep, vloss, color=PALETTE[i], lw=2.5, label=f'{name} (val)')
        axes[1].plot(ep, loss, color=PALETTE[i], lw=1.5, ls='--', alpha=0.6,
                     label=f'{name} (train)')

    for ax, title, ylab in zip(axes,
            ['Model Accuracy', 'Model Loss'], ['Accuracy', 'Loss']):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.legend(fontsize=9,
                  loc='lower right' if 'Acc' in ylab else 'upper right')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Plant Disease Detection -- Transfer Learning Comparison\n'
        'SKNCOE, Pune  |  IJARSCT DOI: 10.48175/IJARSCT-9156',
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "training_history_comparison.png")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  >> Saved: {p}")


def plot_metrics_bar(names, accs, precs, recs, f1s):
    x = np.arange(len(names))
    w = 0.2
    metrics = [accs, precs, recs, f1s]
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (vals, lab, col) in enumerate(zip(metrics, labels, colors)):
        bars = ax.bar(x + i*w, [v*100 for v in vals],
                      w, label=lab, color=col, alpha=0.85, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f'{h:.1f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax.set_xlabel('CNN Architecture', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(
        'Transfer Learning Comparison -- VGG16 vs ResNet50 vs EfficientNetB0\n'
        'Plant Disease Detection | SKNCOE Pune | 2022-2023',
        fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "metrics_comparison_bar.png")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  >> Saved: {p}")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    side = max(8, len(class_names))

    fig, ax = plt.subplots(figsize=(side, side - 2))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title(f'Confusion Matrix -- {model_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  >> Saved: {p}")


def plot_radar_chart(names, accs, precs, recs, f1s):
    cats = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (name, a, p, r, f) in enumerate(zip(names, accs, precs, recs, f1s)):
        vals = [a, p, r, f, a]
        ax.plot(angles, vals, 'o-', lw=2, color=PALETTE[i], label=name)
        ax.fill(angles, vals, alpha=0.15, color=PALETTE[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'], fontsize=8)
    ax.grid(color='grey', ls='--', lw=0.5, alpha=0.7)
    ax.set_title('Model Performance Radar Chart',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "radar_chart_comparison.png")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  >> Saved: {p}")


# ────────────────────────────────────────────────────────────────
#  CSV export
# ────────────────────────────────────────────────────────────────
def save_results_csv(names, accs, precs, recs, f1s, times):
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["Model", "Accuracy (%)", "Precision (%)",
                     "Recall (%)", "F1-Score (%)", "Training Time (min)"])
        for name, a, p, r, f, t in zip(names, accs, precs, recs, f1s, times):
            w.writerow([name, f"{a*100:.2f}", f"{p*100:.2f}",
                        f"{r*100:.2f}", f"{f*100:.2f}", f"{t/60:.1f}"])
    print(f"\n  >> Results saved to: {csv_path}")

    # console summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    hdr = f"{'Model':<22} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Time':>8}"
    print(hdr)
    print("-" * 70)
    for name, a, p, r, f, t in zip(names, accs, precs, recs, f1s, times):
        print(f"{name:<22} {a*100:>7.2f}% {p*100:>7.2f}% "
              f"{r*100:>7.2f}% {f*100:>7.2f}% {t/60:>6.1f}m")
    print("=" * 70)
    best = int(np.argmax(accs))
    print(f"\n  Best model: {names[best]} ({accs[best]*100:.2f}%)")


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  PLANT DISEASE DETECTION -- MODEL COMPARISON STUDY")
    print("  SKNCOE, Savitribai Phule Pune University")
    print("  Author:  Ajinkya Avinash Awari
    print("  Published: IJARSCT DOI 10.48175/IJARSCT-9156")
    print("=" * 60)

    train_gen, val_gen, num_classes = prepare_data()
    class_names = list(val_gen.class_indices.keys())

    models = [
        ("VGG16",          build_vgg16(num_classes)),
        ("ResNet50",       build_resnet50(num_classes)),
        ("EfficientNetB0", build_efficientnet(num_classes)),
    ]

    histories, names = [], []
    acc_l, prec_l, rec_l, f1_l, time_l = [], [], [], [], []

    for name, model in models:
        hist, elapsed = train_model(model, train_gen, val_gen, name)
        acc, prec, rec, f1, yt, yp = evaluate_model(model, val_gen, name)
        plot_confusion_matrix(yt, yp, class_names, name)

        histories.append(hist)
        names.append(name)
        acc_l.append(acc)
        prec_l.append(prec)
        rec_l.append(rec)
        f1_l.append(f1)
        time_l.append(elapsed)

        train_gen.reset()
        val_gen.reset()

    print("\n" + "=" * 60)
    print("  GENERATING COMPARISON GRAPHS")
    print("=" * 60)
    plot_training_history(histories, names)
    plot_metrics_bar(names, acc_l, prec_l, rec_l, f1_l)
    plot_radar_chart(names, acc_l, prec_l, rec_l, f1_l)
    save_results_csv(names, acc_l, prec_l, rec_l, f1_l, time_l)

    print("\n  Done! Check 'results/' for graphs and 'models/' for weights.\n")


if __name__ == "__main__":
    main()
