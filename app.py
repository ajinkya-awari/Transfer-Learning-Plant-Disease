"""
app.py
------
Main application window for Plant Disease Detection.
Lets the user pick a leaf image from disk, choose one of the
three trained models, and see the predicted disease with confidence.

Authors: Ajinkya Awari, Akash Raskar, Shrirameshwar Patil, Namrata Jamdar
Guide:   Prof. Vrushali Paithankar  |  SKNCOE, Pune  |  2022-2023
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json
import threading

import numpy as np
from PIL import Image, ImageTk

from config import (
    MODELS_DIR, RESULTS_DIR, IMAGE_SIZE,
    BG_DARK, BG_CARD, BG_SIDEBAR, ACCENT, ACCENT_HOVER,
    TEXT_WHITE, TEXT_MUTED, BORDER, INPUT_BG,
    FONT_TITLE, FONT_HEAD, FONT_LABEL, FONT_BTN, FONT_SMALL,
)

# defer tensorflow import to keep the window responsive during load
tf = None


def _lazy_import_tf():
    """Import TensorFlow only when we actually need it."""
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf


def load_class_names():
    path = os.path.join(RESULTS_DIR, "class_names.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        mapping = json.load(f)
    return {v: k for k, v in mapping.items()}


class App:
    """Tkinter-based GUI for single-image leaf disease prediction."""

    MODEL_OPTIONS = ["VGG16", "ResNet50", "EfficientNetB0"]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Plant Disease Detection - Predict")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        W, H = 960, 600
        sx = self.root.winfo_screenwidth()
        sy = self.root.winfo_screenheight()
        self.root.geometry(f"{W}x{H}+{(sx-W)//2}+{(sy-H)//2}")

        self.selected_model = tk.StringVar(value=self.MODEL_OPTIONS[2])
        self.img_path = None
        self.photo_ref = None          # keep reference so Tk won't garbage-collect
        self.class_names = load_class_names()

        self._build_sidebar()
        self._build_main_area()

    # ── sidebar ────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=BG_SIDEBAR, width=260)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        tk.Label(sb, text="Plant Disease\nDetection",
                 font=("Segoe UI", 18, "bold"), bg=BG_SIDEBAR,
                 fg=TEXT_WHITE, justify="center").pack(pady=(30, 2))
        tk.Label(sb, text="Transfer Learning Study",
                 font=FONT_SMALL, bg=BG_SIDEBAR,
                 fg=ACCENT).pack()

        tk.Frame(sb, bg=BORDER, height=1, width=220).pack(pady=20)

        # model selector
        tk.Label(sb, text="Select Model", font=FONT_HEAD,
                 bg=BG_SIDEBAR, fg=TEXT_WHITE).pack(anchor="w", padx=20)

        for name in self.MODEL_OPTIONS:
            rb = tk.Radiobutton(
                sb, text=name, variable=self.selected_model,
                value=name, font=FONT_LABEL,
                bg=BG_SIDEBAR, fg=TEXT_MUTED,
                selectcolor=INPUT_BG, activebackground=BG_SIDEBAR,
                activeforeground=ACCENT, indicatoron=True,
                anchor="w",
            )
            rb.pack(fill="x", padx=30, pady=3)

        tk.Frame(sb, bg=BORDER, height=1, width=220).pack(pady=20)

        # open image button
        btn_open = tk.Button(
            sb, text="Open Image", font=FONT_BTN,
            bg=ACCENT, fg=BG_DARK, bd=0, relief="flat",
            cursor="hand2", pady=10,
            command=self._pick_image,
        )
        btn_open.pack(fill="x", padx=20, pady=(0, 10))
        btn_open.bind("<Enter>", lambda _: btn_open.config(bg=ACCENT_HOVER))
        btn_open.bind("<Leave>", lambda _: btn_open.config(bg=ACCENT))

        # predict button
        btn_pred = tk.Button(
            sb, text="Predict Disease", font=FONT_BTN,
            bg="#1565C0", fg=TEXT_WHITE, bd=0, relief="flat",
            cursor="hand2", pady=10,
            command=self._run_prediction,
        )
        btn_pred.pack(fill="x", padx=20)
        btn_pred.bind("<Enter>", lambda _: btn_pred.config(bg="#1976D2"))
        btn_pred.bind("<Leave>", lambda _: btn_pred.config(bg="#1565C0"))

        # bottom credits
        tk.Label(sb, text="SKNCOE, Pune\nIJARSCT DOI: 10.48175/IJARSCT-9156",
                 font=FONT_SMALL, bg=BG_SIDEBAR, fg=TEXT_MUTED,
                 justify="center").pack(side="bottom", pady=14)

    # ── main content area ──────────────────────────────────────
    def _build_main_area(self):
        self.main = tk.Frame(self.root, bg=BG_DARK)
        self.main.pack(side="right", fill="both", expand=True)

        # image preview area (top half)
        self.img_frame = tk.Frame(self.main, bg=BG_CARD, width=680, height=340,
                                  highlightthickness=1,
                                  highlightbackground=BORDER)
        self.img_frame.pack(padx=16, pady=(16, 8), fill="x")
        self.img_frame.pack_propagate(False)

        self.img_label = tk.Label(self.img_frame,
                                  text="No image selected\n\nClick 'Open Image' to load a leaf photo",
                                  font=FONT_LABEL, bg=BG_CARD, fg=TEXT_MUTED)
        self.img_label.pack(expand=True)

        # results area (bottom half)
        self.res_frame = tk.Frame(self.main, bg=BG_CARD, width=680, height=200,
                                  highlightthickness=1,
                                  highlightbackground=BORDER)
        self.res_frame.pack(padx=16, pady=(0, 16), fill="both", expand=True)
        self.res_frame.pack_propagate(False)

        self.res_label = tk.Label(self.res_frame,
                                  text="Prediction results will appear here",
                                  font=FONT_LABEL, bg=BG_CARD, fg=TEXT_MUTED)
        self.res_label.pack(expand=True)

    # ── actions ────────────────────────────────────────────────
    def _pick_image(self):
        ftypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")]
        path = filedialog.askopenfilename(title="Select leaf image",
                                          filetypes=ftypes)
        if not path:
            return

        self.img_path = path

        # display a preview inside img_frame
        pil_img = Image.open(path).convert("RGB")
        # resize for display keeping aspect ratio
        display_w, display_h = 660, 320
        pil_img.thumbnail((display_w, display_h), Image.LANCZOS)
        self.photo_ref = ImageTk.PhotoImage(pil_img)

        self.img_label.config(image=self.photo_ref, text="")

    def _run_prediction(self):
        if self.img_path is None:
            messagebox.showwarning("No image", "Please open an image first.")
            return

        model_name = self.selected_model.get()
        h5_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
        if not os.path.exists(h5_path):
            messagebox.showerror(
                "Model not found",
                f"Could not find {h5_path}\n\n"
                "Run train_comparison.py first to train the models.",
            )
            return

        if self.class_names is None:
            messagebox.showerror(
                "Class map missing",
                "class_names.json not found in results/.\n"
                "Run train_comparison.py first.",
            )
            return

        # show a loading message while TF works
        self.res_label.config(text=f"Loading {model_name} and running inference ...")
        self.root.update_idletasks()

        # run in a thread so the GUI doesn't freeze
        threading.Thread(target=self._predict_thread,
                         args=(model_name, h5_path), daemon=True).start()

    def _predict_thread(self, model_name, h5_path):
        try:
            _lazy_import_tf()
            model = tf.keras.models.load_model(h5_path)

            img = Image.open(self.img_path).convert("RGB")
            img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.array(img_resized) / 255.0
            inp = np.expand_dims(arr, axis=0)

            preds = model.predict(inp, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            confidence = float(preds[top_idx]) * 100
            disease = self.class_names.get(top_idx, f"Class {top_idx}")
            nice_name = disease.replace("___", " - ").replace("_", " ")

            # top 3
            top3 = np.argsort(preds)[::-1][:3]
            lines = []
            for rank, idx in enumerate(top3, 1):
                n = self.class_names.get(int(idx), f"Class {idx}")
                n = n.replace("___", " - ").replace("_", " ")
                lines.append(f"  {rank}.  {n}   ({preds[idx]*100:.1f}%)")

            result_text = (
                f"Model: {model_name}\n\n"
                f"Prediction:  {nice_name}\n"
                f"Confidence:  {confidence:.1f}%\n\n"
                f"Top 3:\n" + "\n".join(lines)
            )

            # update the label back on the main thread
            self.root.after(0, self._show_result, result_text)

        except Exception as exc:
            self.root.after(0, self._show_result, f"Error: {exc}")

    def _show_result(self, text):
        self.res_label.config(text=text, justify="left", anchor="nw",
                              font=("Consolas", 11), padx=20, pady=14)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
