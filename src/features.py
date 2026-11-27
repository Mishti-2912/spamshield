"""Convert cleaned text into numbers using TF-IDF.

Machine-learning models only understand numbers, not words. TF-IDF stands for
*Term Frequency - Inverse Document Frequency*. In simple terms:

  - TF (Term Frequency): how often a word appears in a message.
  - IDF (Inverse Document Frequency): how rare a word is across all messages.
    Common words get a low weight; rare words get a high weight.

So a word like 'winner' or 'free' that shows up a lot in spam but rarely in
legitimate messages ends up with a high TF-IDF score, which is exactly the
signal the model needs to spot spam.

This module wraps scikit-learn's TfidfVectorizer so the project has one
consistent place that defines how text becomes numbers.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features=5000, ngram_range=(1, 2)):
    """Create a TF-IDF vectorizer.

    Parameters
    ----------
    max_features : keep only the top N most important words/bigrams.
    ngram_range   : (1, 2) means we use single words AND pairs of words
                    (e.g. 'free', 'free entry'). Pairs capture phrases like
                    'claim prize' that single words miss.
    """
    return TfidfVectorizer(
        lowercase=False,                # we already lowercased in preprocessing
        stop_words=None,                # we already removed stop words
        token_pattern=r"(?u)\b\w+\b",   # a token is any run of word characters
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,              # use 1 + log(tf), dampens very frequent words
    )


def fit_transform(vectorizer, texts):
    """Learn the vocabulary from `texts` and return the TF-IDF matrix."""
    return vectorizer.fit_transform(texts)


def transform(vectorizer, texts):
    """Convert new text into a TF-IDF matrix using an already-trained vectorizer."""
    return vectorizer.transform(texts)
