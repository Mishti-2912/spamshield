"""Load and prepare the spam/ham dataset.

This module is responsible for getting the raw data into a clean pandas
DataFrame that the rest of the project can use. It will:

  1. Try to read a local CSV file (`data/spam.csv`).
  2. If the file is missing, download it automatically (first run only).

Beginner note
-------------
A *DataFrame* is just a table (like an Excel sheet) that pandas gives us so we
can filter, count, and transform rows easily. Here it has two columns:

    label    : 'ham' (legitimate) or 'spam' (junk / phishing)
    message  : the actual text of the message
"""

import os

import pandas as pd

# Paths ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOCAL_CSV = os.path.join(DATA_DIR, "spam.csv")

# Public mirror of the SMS Spam Collection dataset (tab-separated).
# Column 1 = label (ham/spam), Column 2 = the message text.
DATASET_URL = "x"


def _download_and_save(url: str = DATASET_URL, output_path: str = LOCAL_CSV) -> pd.DataFrame:
    """Download the raw dataset and save it as a clean CSV (used on first run)."""
    df = pd.read_csv(
        url, sep="\t", header=None,
        names=["label", "message"], encoding="utf-8",
    )
    df = df[["label", "message"]].dropna()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[data_loader] Downloaded and saved dataset to {output_path}")
    return df


def load_data(path: str = LOCAL_CSV) -> pd.DataFrame:
    """Load the dataset as a DataFrame with columns ['label', 'message'].

    If the local file does not exist yet, it is downloaded automatically so the
    project works on a brand-new computer without extra setup.
    """
    if not os.path.exists(path):
        print("[data_loader] Local dataset not found. Downloading it now...")
        df = _download_and_save(output_path=path)
    else:
        # Try utf-8 first, fall back to latin-1 (some copies of this dataset
        # use a different encoding).
        df = None
        for encoding in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"Could not read {path} with utf-8 or latin-1 encoding.")

    df = _standardize_columns(df)
    df = df.dropna(subset=["message"]).reset_index(drop=True)
    df["message"] = df["message"].astype(str)
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the DataFrame has exactly 'label' and 'message' columns."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # If the file had no header, take the first two columns and rename them.
    if "label" not in df.columns or "message" not in df.columns:
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]

    # Keep only standard spam/ham labels and tidy them up.
    df = df[df["label"].astype(str).str.lower().isin(["ham", "spam"])]
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df[["label", "message"]]


def add_numeric_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'label_num' column: ham = 0 (legitimate), spam = 1.

    Models need numbers, not words, so we map the two labels to 0 and 1.
    """
    df = df.copy()
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def label_name(label_num) -> str:
    """Return a friendly name for a numeric label (0 -> Legitimate, 1 -> Spam)."""
    return "Spam" if int(label_num) == 1 else "Legitimate"
