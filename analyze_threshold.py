"""Examine the precision-recall tradeoff and justify the spam cutoff.

By default a classifier says 'Spam' when P(spam) > 0.5. We can lower that
cutoff to catch more spam (higher recall) at the cost of a few extra false
alarms (lower precision). This script prints precision, recall and F1 at
several cutoffs using 5-fold cross-validation on the TRAINING set only, so the
test set is never used for tuning. It then reports the chosen cutoff and the
final performance on the untouched test set.

Usage:
    python analyze_threshold.py
"""

import os

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.data_loader import add_numeric_label, load_data
from src.features import build_vectorizer, fit_transform
from src.model import find_best_threshold, load_model, train_model
from src.preprocessing import apply_cleaning

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(PROJECT_ROOT, "models")


def main():
    df = add_numeric_label(apply_cleaning(load_data()))
    vectorizer = build_vectorizer()
    X = fit_transform(vectorizer, df["clean_message"])
    y = df["label_num"].values

    # 80/20 split; the test set is held back until the very end.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Out-of-fold spam probabilities on the TRAINING set (honest estimate).
    oof = cross_val_predict(
        MultinomialNB(), X_train, y_train, cv=5, method="predict_proba",
    )[:, 1]

    print("Precision-recall tradeoff across cutoffs (5-fold CV on train set):")
    print(f"{'Cutoff':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    for t in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
        pred = (oof >= t).astype(int)
        p = precision_score(y_train, pred, pos_label=1, zero_division=0)
        r = recall_score(y_train, pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_train, pred, pos_label=1, zero_division=0)
        print(f"{t:>8.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

    best_t = find_best_threshold(y_train, oof)
    print(f"\nChosen cutoff (max F1 on train CV): {best_t:.2f}")

    # Final, fair evaluation on the untouched test set.
    model, _ = load_model(MODELS)
    if model is None:
        model = train_model(X_train, y_train)
    test_probs = model.predict_proba(X_test)[:, 1]
    pred = (test_probs >= best_t).astype(int)
    p = precision_score(y_test, pred, pos_label=1, zero_division=0)
    r = recall_score(y_test, pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, pred, pos_label=1, zero_division=0)
    print("\nFinal performance on the TEST set at the chosen cutoff:")
    print(f"  Precision: {p:.4f}   Recall: {r:.4f}   F1: {f1:.4f}")


if __name__ == "__main__":
    main()
