"""Clean raw message text.

Before we can feed text into a machine-learning model, we clean it so the
model focuses on meaningful words instead of noise (links, symbols, etc.).

Steps applied to every message:
  1. Convert to lowercase.
  2. Remove HTML tags, web links, and email addresses.
  3. Remove everything that is not a letter, digit, or space (keep numbers,
     because prize amounts and codes like "1000" are useful spam signals).
  4. Collapse multiple spaces into one.
  5. Remove common English 'stop words' (the, and, of, ...) that carry little
     meaning and appear in both spam and ham.

The cleaned text is stored in a new column called 'clean_message'.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Pre-compile the regex patterns once (faster than recompiling each time).
_html_re = re.compile(r"<[^>]+>")
_url_re = re.compile(r"https?://\S+|www\.\S+")
_email_re = re.compile(r"\S+@\S+")
_non_alpha_re = re.compile(r"[^a-z0-9\s]")
_multi_space_re = re.compile(r"\s+")

# Convert the frozenset of stop words into a Python set for fast lookups.
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Clean a single message string and return the cleaned text."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML tags, URLs, and email addresses
    text = _html_re.sub(" ", text)
    text = _url_re.sub(" ", text)
    text = _email_re.sub(" ", text)
    # 3. Keep only letters and spaces
    text = _non_alpha_re.sub(" ", text)
    # 4. Collapse multiple spaces into one and trim the ends
    text = _multi_space_re.sub(" ", text).strip()
    # 5. Remove stop words
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def apply_cleaning(df, text_col: str = "message", out_col: str = "clean_message"):
    """Add a cleaned-text column to the DataFrame."""
    df = df.copy()
    df[out_col] = df[text_col].apply(clean_text)
    # If a cleaned message becomes empty after stop-word removal, use a tiny
    # placeholder so the vectorizer never receives an empty string.
    df[out_col] = df[out_col].replace("", "empty")
    return df
