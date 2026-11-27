"""Measure how well the model performs.

We evaluate a classifier with several metrics (all explained simply):

  - Accuracy : out of ALL messages, what fraction did we classify correctly?
               Gives the overall picture, but can be misleading when the
               classes are unbalanced (we have more ham than spam).

  - Precision (for spam): of all messages we CALLED spam, how many really were?
               High precision = few false alarms (few real messages wrongly
               flagged as spam).

  - Recall (for spam): of all messages that really ARE spam, how many did we
               catch? High recall = we miss few spam messages.

  - F1-score : a balance between precision and recall (their harmonic mean).
               Use this when the classes are unbalanced.

  - Confusion matrix : a 2x2 table of correct and wrong predictions:
      [True Legitimate,   False Spam     ]
      [False Legitimate,  True Spam      ]

The decision `threshold` (spam cutoff) decides how aggressive the filter is.
Lowering it catches more spam (higher recall) but causes a few extra false
alarms (lower precision). See model.tune_threshold_cv for how it is chosen.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .eda import plot_confusion_matrix


def evaluate_model(model, X_test_tfidf, y_test, threshold=0.5):
    """Run predictions and return a dictionary with all metrics + artifacts.

    A message is predicted as spam when its P(spam) is >= `threshold`.
    """
    # Probability of the spam class (column index 1) for every test message.
    spam_probs = model.predict_proba(X_test_tfidf)[:, 1]
    y_pred = (spam_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Legitimate (ham)", "Spam"],
        digits=4, zero_division=0,
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }


def format_metrics(metrics):
    """Return a nicely formatted multi-line string of the key metrics."""
    cm = metrics["confusion_matrix"]
    lines = [
        "=" * 50,
        "            MODEL EVALUATION RESULTS",
        "=" * 50,
        f"Spam decision threshold : {metrics['threshold']:.2f}",
        f"Accuracy : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall   : {metrics['recall']:.4f}",
        f"F1-score : {metrics['f1']:.4f}",
        "-" * 50,
        "Confusion Matrix (rows = true, columns = predicted):",
        f"  Legitimate -> {cm[0]}",
        f"  Spam       -> {cm[1]}",
        "-" * 50,
        "Classification Report:",
        metrics["classification_report"],
        "=" * 50,
    ]
    return "\n".join(lines)


def save_evaluation_plot(metrics, output_path):
    """Save the confusion-matrix heatmap to a PNG file."""
    return plot_confusion_matrix(metrics["confusion_matrix"], save_path=output_path)
