"""Download the SMS Spam Collection dataset.

This script fetches the SMS Spam Collection dataset (a public, free dataset
used widely for spam-detection research) and saves it as a clean CSV file
named `spam.csv` inside the `data/` folder.

If you are offline or the download fails, you can also place any CSV with two
columns named `label` and `message` (where label is 'ham' or 'spam') into the
`data/` folder and rename it to `spam.csv`. The rest of the project will still
work.

Usage (from the project root):
    python data/download_dataset.py
"""

import os
import sys

import pandas as pd

# Public mirror of the SMS Spam Collection dataset (tab-separated).
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "spam.csv")


def download_dataset(url: str = DATASET_URL, output_path: str = OUTPUT_PATH) -> str:
    """Download the dataset and save it as a clean CSV. Returns the output path."""
    print(f"Downloading dataset from:\n  {url}")
    try:
        df = pd.read_csv(
            url, sep="\t", header=None,
            names=["label", "message"], encoding="utf-8",
        )
    except Exception as exc:
        print("\nERROR: Could not download the dataset.")
        print("Please check your internet connection, then try again.")
        print(f"Details: {exc}")
        sys.exit(1)

    df = df[["label", "message"]].dropna()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nSaved {len(df):,} rows to:\n  {output_path}")
    print(f"Spam messages : {len(df[df['label'] == 'spam']):,}")
    print(f"Ham  messages : {len(df[df['label'] == 'ham']):,}")
    return output_path


if __name__ == "__main__":
    download_dataset()
