"""
registration.py
---------------
New-user registration form for the Plant Disease Detection app.
Validates inputs and stores credentials in the local SQLite database.

Author:  Ajinkya Avinash Awari
Guide:   Prof. Vrushali Paithankar  |  SKNCOE, Pune  |  SKNCOE, Pune
"""

import tkinter as tk
from tkinter import messagebox
import sqlite3
import re
import os
import sys

from config import (
    DB_PATH, BG_DARK, BG_CARD, ACCENT, ACCENT_HOVER, ACCENT_RED, RED_HOVER,
    TEXT_WHITE, TEXT_MUTED, BORDER, INPUT_BG,
    FONT_TITLE, FONT_LABEL, FONT_ENTRY, FONT_BTN, FONT_SMALL,
)


def init_db():
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


def register_user(root, fields):
    """Validate all fields and insert a new row into the database."""
    name  = fields["fullname"].get().strip()
    email = fields["email"].get().strip()
    uname = fields["username"].get().strip()
    pwd   = fields["password"].get().strip()
    cpwd  = fields["confirm"].get().strip()

    # basic checks
    if not all([name, email, uname, pwd, cpwd]):
        messagebox.showwarning("Missing fields", "Please fill in every field.")
        return

    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        messagebox.showwarning("Invalid email",
                               "Please enter a valid email address.")
        return

    if len(uname) < 3:
        messagebox.showwarning("Username too short",
                               "Username must be at least 3 characters.")
        return

    if len(pwd) < 6:
        messagebox.showwarning("Weak password",
                               "Password must be at least 6 characters.")
        return

    if pwd != cpwd:
        messagebox.showerror("Mismatch", "Passwords do not match.")
        return

    # try to insert
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO users (fullname, email, username, password) "
            "VALUES (?, ?, ?, ?)",
            (name, email, uname, pwd),
        )
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        messagebox.showerror("Already exists",
                             "That username or email is already registered.")
        return
    except Exception as exc:
        messagebox.showerror("Database error", str(exc))
        return

    messagebox.showinfo("Success",
                        f"Account created for {name}!\nYou can now log in.")
    go_to_login(root)


def go_to_login(root):
    root.destroy()
    os.system(f'"{sys.executable}" login.py')


def _make_field(parent, label_text, show=""):
    """Reusable label + entry widget. Returns the StringVar."""
    tk.Label(parent, text=label_text, font=FONT_LABEL,
             bg=BG_CARD, fg=TEXT_MUTED, anchor="w").pack(anchor="w", padx=40)
    var = tk.StringVar()
    frame = tk.Frame(parent, bg=BORDER, highlightthickness=1,
                     highlightbackground=BORDER)
    frame.pack(fill="x", padx=40, pady=(3, 10))
    entry = tk.Entry(frame, textvariable=var, font=FONT_ENTRY,
                     bg=INPUT_BG, fg=TEXT_WHITE, bd=0,
                     insertbackground=ACCENT, relief="flat")
    if show:
        entry.config(show=show)
    entry.pack(padx=10, pady=8, fill="x")
    return var


def build_window():
    init_db()

    root = tk.Tk()
    root.title("Plant Disease Detection - Register")
    root.resizable(False, False)
    root.configure(bg=BG_DARK)

    W, H = 520, 640
    sx, sy = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(sx-W)//2}+{(sy-H)//2}")

    card = tk.Frame(root, bg=BG_CARD,
                    highlightthickness=1, highlightbackground=BORDER)
    card.place(relx=0.5, rely=0.5, anchor="center", width=460, height=600)

    tk.Label(card, text="Create Account", font=FONT_TITLE,
             bg=BG_CARD, fg=TEXT_WHITE).pack(pady=(24, 2))
    tk.Label(card, text="Fill in your details below",
             font=("Segoe UI", 11), bg=BG_CARD, fg=TEXT_MUTED).pack()

    tk.Frame(card, bg=BORDER, height=1, width=380).pack(pady=16)

    fields = {}
    fields["fullname"] = _make_field(card, "Full Name")
    fields["email"]    = _make_field(card, "Email")
    fields["username"] = _make_field(card, "Username")
    fields["password"] = _make_field(card, "Password", show="*")
    fields["confirm"]  = _make_field(card, "Confirm Password", show="*")

    # register button
    btn_reg = tk.Button(card, text="Register", font=FONT_BTN,
                        bg=ACCENT, fg=BG_DARK, bd=0, relief="flat",
                        cursor="hand2", padx=20, pady=10,
                        command=lambda: register_user(root, fields))
    btn_reg.pack(fill="x", padx=40, pady=(8, 10))
    btn_reg.bind("<Enter>", lambda _: btn_reg.config(bg=ACCENT_HOVER))
    btn_reg.bind("<Leave>", lambda _: btn_reg.config(bg=ACCENT))

    # back to login
    btn_back = tk.Button(card, text="Back to Login",
                         font=("Segoe UI", 10), bg=BG_CARD, fg=ACCENT,
                         bd=1, relief="solid", cursor="hand2", pady=7,
                         command=lambda: go_to_login(root))
    btn_back.pack(fill="x", padx=40)

    root.mainloop()


if __name__ == "__main__":
    build_window()
