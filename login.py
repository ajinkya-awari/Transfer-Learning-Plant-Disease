"""
login.py
--------
Login window for the Plant Disease Detection desktop app.
Authenticates against a local SQLite database and launches
the main application on success.

Authors: Ajinkya Awari, Akash Raskar, Shrirameshwar Patil, Namrata Jamdar
Guide:   Prof. Vrushali Paithankar  |  SKNCOE, Pune  |  2022-2023
"""

import tkinter as tk
from tkinter import messagebox
import sqlite3
import os
import sys

from config import (
    DB_PATH, BG_DARK, BG_CARD, ACCENT, ACCENT_HOVER,
    TEXT_WHITE, TEXT_MUTED, BORDER, INPUT_BG,
    FONT_TITLE, FONT_LABEL, FONT_ENTRY, FONT_BTN, FONT_SMALL,
)


def init_db():
    """Create the users table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            email    TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def attempt_login(root, user_var, pass_var):
    """Check credentials and open the main app if valid."""
    uname = user_var.get().strip()
    pwd   = pass_var.get().strip()

    if not uname or not pwd:
        messagebox.showwarning("Missing fields",
                               "Enter both username and password.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT fullname FROM users WHERE username=? AND password=?",
                    (uname, pwd))
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        messagebox.showerror("Database error", str(exc))
        return

    if row:
        messagebox.showinfo("Welcome", f"Login successful!\nHello, {row[0]}")
        root.destroy()
        # launch the main prediction GUI
        os.system(f'"{sys.executable}" app.py')
    else:
        messagebox.showerror("Login failed",
                             "Incorrect username or password.")


def open_register(root):
    """Close login and open the registration window."""
    root.destroy()
    os.system(f'"{sys.executable}" registration.py')


def _make_field(parent, label_text, show_char=""):
    """Helper: label + styled entry field. Returns the StringVar."""
    tk.Label(parent, text=label_text, font=FONT_LABEL,
             bg=BG_CARD, fg=TEXT_MUTED, anchor="w").pack(anchor="w", padx=34)
    var = tk.StringVar()
    frame = tk.Frame(parent, bg=BORDER, highlightthickness=1,
                     highlightbackground=BORDER)
    frame.pack(fill="x", padx=34, pady=(4, 16))
    entry = tk.Entry(frame, textvariable=var, font=FONT_ENTRY,
                     bg=INPUT_BG, fg=TEXT_WHITE, bd=0,
                     insertbackground=ACCENT, relief="flat")
    if show_char:
        entry.config(show=show_char)
    entry.pack(padx=10, pady=10, fill="x")
    return var


def build_window():
    init_db()

    root = tk.Tk()
    root.title("Plant Disease Detection - Login")
    root.resizable(False, False)
    root.configure(bg=BG_DARK)

    W, H = 920, 560
    sx, sy = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(sx-W)//2}+{(sy-H)//2}")

    # ── left branding panel ────────────────────────────────────
    left = tk.Frame(root, bg=BG_CARD, width=400)
    left.pack(side="left", fill="y")
    left.pack_propagate(False)

    # vertical centering spacer
    tk.Frame(left, bg=BG_CARD, height=90).pack()

    tk.Label(left, text="Plant Disease\nDetection",
             font=("Segoe UI", 24, "bold"), bg=BG_CARD,
             fg=TEXT_WHITE, justify="center").pack(pady=(0, 4))
    tk.Label(left, text="using Transfer Learning",
             font=("Segoe UI", 12), bg=BG_CARD,
             fg=ACCENT).pack()

    # thin separator line
    tk.Frame(left, bg=BORDER, height=1, width=260).pack(pady=28)

    tk.Label(left, text="VGG16   |   ResNet50   |   EfficientNetB0",
             font=("Segoe UI", 9), bg=BG_CARD,
             fg=TEXT_MUTED).pack()
    tk.Label(left, text="PlantVillage Dataset  -  38 disease classes",
             font=("Segoe UI", 9), bg=BG_CARD,
             fg=TEXT_MUTED).pack(pady=(4, 0))

    tk.Frame(left, bg=BORDER, height=1, width=260).pack(pady=28)

    tk.Label(left, text="Published: IJARSCT 2023",
             font=("Segoe UI", 9, "italic"), bg=BG_CARD,
             fg=TEXT_MUTED).pack()
    tk.Label(left, text="DOI: 10.48175/IJARSCT-9156",
             font=FONT_SMALL, bg=BG_CARD, fg=TEXT_MUTED).pack(pady=(2, 0))

    # bottom credit
    tk.Label(left, text="SKNCOE, Savitribai Phule Pune University",
             font=FONT_SMALL, bg=BG_CARD, fg=TEXT_MUTED,
             justify="center").pack(side="bottom", pady=18)

    # ── right login card ───────────────────────────────────────
    right = tk.Frame(root, bg=BG_DARK)
    right.pack(side="right", fill="both", expand=True)

    card = tk.Frame(right, bg=BG_CARD,
                    highlightthickness=1, highlightbackground=BORDER)
    card.place(relx=0.5, rely=0.5, anchor="center", width=400, height=440)

    tk.Label(card, text="Welcome Back", font=FONT_TITLE,
             bg=BG_CARD, fg=TEXT_WHITE).pack(pady=(30, 2))
    tk.Label(card, text="Sign in to continue",
             font=("Segoe UI", 11), bg=BG_CARD, fg=TEXT_MUTED).pack()

    tk.Frame(card, bg=BORDER, height=1, width=330).pack(pady=22)

    user_var = _make_field(card, "Username")
    pass_var = _make_field(card, "Password", show_char="*")

    # enter key shortcut
    root.bind("<Return>", lambda _: attempt_login(root, user_var, pass_var))

    # sign in button
    btn = tk.Button(card, text="Sign In", font=FONT_BTN,
                    bg=ACCENT, fg=BG_DARK, bd=0, relief="flat",
                    cursor="hand2", padx=20, pady=10,
                    command=lambda: attempt_login(root, user_var, pass_var))
    btn.pack(fill="x", padx=34, pady=(4, 12))
    btn.bind("<Enter>", lambda _: btn.config(bg=ACCENT_HOVER))
    btn.bind("<Leave>", lambda _: btn.config(bg=ACCENT))

    # register link
    reg = tk.Button(card, text="Create New Account",
                    font=("Segoe UI", 10), bg=BG_CARD, fg=ACCENT,
                    bd=1, relief="solid", cursor="hand2", pady=8,
                    command=lambda: open_register(root))
    reg.pack(fill="x", padx=34)

    root.mainloop()


if __name__ == "__main__":
    build_window()
