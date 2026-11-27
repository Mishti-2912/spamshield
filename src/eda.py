"""Exploratory Data Analysis (EDA) and visualizations.

EDA means *getting to know your data* before building the model. We look at
how many spam vs ham messages there are, how long they are, and which words
appear most often in each group.

Every plotting function returns a matplotlib Figure. If `save_path` is given
the figure is saved to that file; otherwise the figure is returned so it can
be shown directly (e.g. in a notebook or Streamlit with st.pyplot).
"""

import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Use a non-interactive backend so plots can be saved without a display.
matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

# Consistent colours used across all charts.
HAM_COLOR = "#2e86de"     # blue  -> legitimate
SPAM_COLOR = "#e55039"   # red   -> spam
PALETTE = {"ham": HAM_COLOR, "spam": SPAM_COLOR}


def _word_counts(text_series, top_n=20):
    """Return the most common words and their counts from a series of text."""
    all_words = " ".join(text_series).split()
    counter = Counter(all_words)
    return counter.most_common(top_n)


def plot_class_distribution(df, save_path=None):
    """Bar chart showing the number of ham vs spam messages."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="label", hue="label", palette=PALETTE,
                  ax=ax, legend=False)
    ax.set_title("Number of Legitimate (ham) vs Spam Messages", fontsize=13)
    ax.set_xlabel("Message Type")
    ax.set_ylabel("Count")
    counts = df["label"].value_counts()
    for i, label in enumerate(counts.index):
        ax.text(i, counts[label] + 30, f"{counts[label]:,}",
                ha="center", fontweight="bold")
    plt.tight_layout()
    return _maybe_save(fig, save_path)


def plot_message_length_distribution(df, save_path=None):
    """Histogram of message character length, split by label."""
    df = df.copy()
    df["length"] = df["message"].str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="length", hue="label", bins=50,
                 palette=PALETTE, ax=ax)
    ax.set_title("Distribution of Message Length", fontsize=13)
    ax.set_xlabel("Number of Characters")
    ax.set_ylabel("Number of Messages")
    plt.tight_layout()
    return _maybe_save(fig, save_path)


def plot_word_count_distribution(df, save_path=None):
    """Histogram of word count per message, split by label."""
    df = df.copy()
    df["word_count"] = df["clean_message"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="word_count", hue="label", bins=40,
                 palette=PALETTE, ax=ax)
    ax.set_title("Distribution of Word Count per Message", fontsize=13)
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Number of Messages")
    plt.tight_layout()
    return _maybe_save(fig, save_path)


def plot_top_words(df, label="spam", top_n=20, save_path=None):
    """Horizontal bar chart of the most common words in spam or ham messages."""
    text = df[df["label"] == label]["clean_message"]
    common = _word_counts(text, top_n=top_n)
    if not common:
        return None
    words, counts = zip(*common)
    color = SPAM_COLOR if label == "spam" else HAM_COLOR
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=list(counts), y=list(words), color=color, ax=ax)
    title = ("Top Words in Spam Messages"
             if label == "spam" else "Top Words in Legitimate Messages")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    plt.tight_layout()
    return _maybe_save(fig, save_path)


def plot_confusion_matrix(cm, labels=("Legitimate", "Spam"), save_path=None):
    """Heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return _maybe_save(fig, save_path)


def _maybe_save(fig, save_path):
    """Either save the figure to a file or return it for display."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig


def generate_all_plots(df, output_dir):
    """Create and save every EDA plot plus the top-word charts. Used by main.py."""
    os.makedirs(output_dir, exist_ok=True)
    plot_class_distribution(df, save_path=os.path.join(output_dir, "01_class_distribution.png"))
    plot_message_length_distribution(df, save_path=os.path.join(output_dir, "02_message_length.png"))
    plot_word_count_distribution(df, save_path=os.path.join(output_dir, "03_word_count.png"))
    plot_top_words(df, label="spam", save_path=os.path.join(output_dir, "04_top_words_spam.png"))
    plot_top_words(df, label="ham", save_path=os.path.join(output_dir, "05_top_words_ham.png"))
    print(f"[eda] Saved all EDA plots to {output_dir}")


def dataset_summary(df):
    """Return a dictionary of key statistics for the dashboard/report."""
    df = df.copy()
    df["length"] = df["message"].str.len()
    df["word_count"] = df["clean_message"].str.split().str.len()
    counts = df["label"].value_counts()
    total = len(df)
    return {
        "total_messages": total,
        "spam_count": int(counts.get("spam", 0)),
        "ham_count": int(counts.get("ham", 0)),
        "spam_percent": round(100 * counts.get("spam", 0) / total, 2),
        "ham_percent": round(100 * counts.get("ham", 0) / total, 2),
        "avg_length": round(df["length"].mean(), 1),
        "avg_words": round(df["word_count"].mean(), 1),
        "avg_length_spam": round(df[df["label"] == "spam"]["length"].mean(), 1),
        "avg_length_ham": round(df[df["label"] == "ham"]["length"].mean(), 1),
    }
