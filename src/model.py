"""Train, tune, and save the Multinomial Naive Bayes classifier.

Why Multinomial Naive Bayes?
  - It is designed for text/word-count features (exactly what TF-IDF gives us).
  - It is fast, simple, and works very well for spam detection.
  - It is based on Bayes' theorem from probability theory, which fits this
    Probability & Statistics project perfectly.

The idea (in one line)
---------------------
Naive Bayes calculates P(spam | message) and P(ham | message) using Bayes'
theorem, then picks the label with the higher probability. 'Naive' means it
assumes words appear independently of each other - a simplification that still
works surprisingly well in practice.

    P(spam | message) proportional to P(spam) * P(message | spam)

Tuning the spam cutoff (the precision-recall tradeoff)
-----------------------------------------------------
By default a message is called Spam when P(spam) > 0.5. For a spam/phishing
filter, *missing* spam is usually worse than a few false alarms, so we let the
data choose a better cutoff using cross-validation on the TRAINING set only
(see `tune_threshold_cv`). The test set is never used for tuning, so it stays a
fair, unbiased measure of performance.
"""

import json
import os

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# Fallback cutoff used when no tuned threshold has been computed yet.
DEFAULT_SPAM_THRESHOLD = 0.5


def build_model(alpha=1.0):
    """Create a Multinomial Naive Bayes model.

    `alpha` is a 'smoothing' value. When a word never appeared in spam during
    training, the model would otherwise give it a probability of 0 and multiply
    everything to zero. Smoothing (alpha=1, Laplace smoothing) adds a small
    count to every word so probabilities never hit zero.
    """
    return MultinomialNB(alpha=alpha)


def build_pipeline(max_features=5000, ngram_range=(1, 2), alpha=1.0):
    """A single object that does TF-IDF + Naive Bayes together.

    Handy because you can save/load one object and call predict() directly on
    raw cleaned text without redoing the vectorizer step manually.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            stop_words=None,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )),
        ("nb", MultinomialNB(alpha=alpha)),
    ])


def train_model(X_train_tfidf, y_train, alpha=1.0):
    """Train Naive Bayes on already-vectorized features and return the model."""
    model = build_model(alpha=alpha)
    model.fit(X_train_tfidf, y_train)
    return model


def find_best_threshold(y_true, spam_probs):
    """Return the spam-probability cutoff that maximises the F1 score.

    Parameters
    ----------
    y_true      : true labels (0 = ham, 1 = spam).
    spam_probs   : the model's P(spam) for those same rows.

    We test many cutoffs between 0.05 and 0.50 and keep the one with the best
    F1 (a balance of precision and recall).
    """
    y_true = np.asarray(y_true)
    spam_probs = np.asarray(spam_probs)
    best_t, best_f1 = DEFAULT_SPAM_THRESHOLD, -1.0
    for t in np.linspace(0.05, 0.50, 46):
        preds = (spam_probs >= t).astype(int)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def tune_threshold_cv(X_train, y_train, alpha=1.0, cv=5):
    """Pick the best spam cutoff using cross-validation on the training set.

    We never touch the test set here, so the test set stays a fair, unbiased
    measure of performance. `cross_val_predict` gives 'out-of-fold' spam
    probabilities: each row is predicted by a model that did NOT see that row
    during training (an honest estimate). Returns the chosen threshold.
    """
    oof_probs = cross_val_predict(
        MultinomialNB(alpha=alpha), X_train, y_train,
        cv=cv, method="predict_proba",
    )[:, 1]
    return find_best_threshold(y_train, oof_probs)


def save_model(model, vectorizer, models_dir, threshold=None, filename_prefix="spamshield"):
    """Save the model, vectorizer, and the tuned decision threshold."""
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{filename_prefix}_model.joblib")
    vec_path = os.path.join(models_dir, f"{filename_prefix}_vectorizer.joblib")
    meta_path = os.path.join(models_dir, f"{filename_prefix}_meta.json")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    with open(meta_path, "w") as f:
        json.dump({"threshold": threshold}, f)
    print(f"[model] Saved model      -> {model_path}")
    print(f"[model] Saved vectorizer -> {vec_path}")
    print(f"[model] Saved threshold   -> {meta_path} (value={threshold})")
    return model_path, vec_path


def load_model(models_dir, filename_prefix="spamshield"):
    """Load a previously saved model and vectorizer. Returns (None, None) if missing."""
    model_path = os.path.join(models_dir, f"{filename_prefix}_model.joblib")
    vec_path = os.path.join(models_dir, f"{filename_prefix}_vectorizer.joblib")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        return None, None
    return joblib.load(model_path), joblib.load(vec_path)


def load_threshold(models_dir, filename_prefix="spamshield"):
    """Load the saved decision threshold, or fall back to the default 0.5."""
    meta_path = os.path.join(models_dir, f"{filename_prefix}_meta.json")
    if not os.path.exists(meta_path):
        return DEFAULT_SPAM_THRESHOLD
    with open(meta_path) as f:
        return float(json.load(f).get("threshold", DEFAULT_SPAM_THRESHOLD))
