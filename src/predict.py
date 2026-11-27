"""Classify a single message as Spam or Legitimate.

This is the 'live prediction' feature. Given a raw email/message and a trained
model + vectorizer, it cleans the text, converts it to TF-IDF numbers, and
asks the model for the probability it is spam. A message is Spam when that
probability reaches the `threshold` (chosen during training to balance
precision and recall).
"""

from .data_loader import label_name
from .preprocessing import clean_text


def predict_message(model, vectorizer, message, threshold=0.5):
    """Predict whether a single message is spam.

    Returns a dictionary with:
      - label         : 0 (legitimate) or 1 (spam)
      - label_name    : 'Legitimate' or 'Spam'
      - spam_prob     : probability the message is spam (0 to 1)
      - ham_prob      : probability the message is legitimate (0 to 1)
      - threshold     : the cutoff used to make the decision
      - clean_message : the cleaned text the model actually saw
    """
    cleaned = clean_text(message)
    if cleaned.strip() == "":
        cleaned = "empty"

    # The vectorizer expects an iterable of strings.
    vector = vectorizer.transform([cleaned])
    # predict_proba returns one row; columns follow class order [0, 1] which
    # for our labels is [ham, spam].
    probs = model.predict_proba(vector)[0]
    ham_prob = float(probs[0])
    spam_prob = float(probs[1])

    # Apply the tuned decision threshold instead of the default 0.5.
    label = 1 if spam_prob >= threshold else 0

    return {
        "label": label,
        "label_name": label_name(label),
        "spam_prob": spam_prob,
        "ham_prob": ham_prob,
        "threshold": float(threshold),
        "clean_message": cleaned,
    }
