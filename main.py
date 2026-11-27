"""SpamShield - full training pipeline.

Run this script once to train and evaluate the model end to end:

    load data -> clean text -> EDA (save plots) -> TF-IDF -> split ->
    train Naive Bayes -> tune the spam cutoff -> evaluate -> save model

Usage (from the project root folder):
    python main.py

Outputs
-------
  data/spam.csv            the dataset (downloaded if missing)
  outputs/*.png            all EDA and confusion-matrix plots
  models/*.joblib          saved model + vectorizer
  models/spamshield_meta.json   the tuned spam decision threshold
  outputs/metrics.txt      text report of the metrics
"""

import os

from sklearn.model_selection import train_test_split

from src.data_loader import add_numeric_label, load_data
from src.eda import dataset_summary, generate_all_plots
from src.evaluate import evaluate_model, format_metrics, save_evaluation_plot
from src.features import build_vectorizer, fit_transform
from src.model import save_model, train_model, tune_threshold_cv
from src.preprocessing import apply_cleaning

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def main():
    print("=" * 55)
    print("   SpamShield - Spam & Phishing Email Classifier")
    print("   Multinomial Naive Bayes + TF-IDF")
    print("=" * 55)

    # 1. Load (and download on first run) the dataset.
    print("\n[1/8] Loading dataset...")
    df = load_data()
    df = add_numeric_label(df)
    counts = df["label"].value_counts().to_dict()
    print(f"      Loaded {len(df):,} messages ({counts})")

    # 2. Clean the text.
    print("\n[2/8] Cleaning text...")
    df = apply_cleaning(df)
    print("      Example cleaned message:")
    print(f"      Original : {df['message'].iloc[0][:70]}...")
    print(f"      Cleaned  : {df['clean_message'].iloc[0][:70]}...")

    # 3. Exploratory Data Analysis - save plots to outputs/.
    print("\n[3/8] Running EDA and saving plots...")
    generate_all_plots(df, OUTPUT_DIR)
    summary = dataset_summary(df)
    print(f"      Spam: {summary['spam_count']} ({summary['spam_percent']}%) | "
          f"Legitimate: {summary['ham_count']} ({summary['ham_percent']}%)")
    print(f"      Avg length - spam: {summary['avg_length_spam']} chars, "
          f"ham: {summary['avg_length_ham']} chars")

    # 4. TF-IDF feature extraction.
    print("\n[4/8] Extracting TF-IDF features...")
    vectorizer = build_vectorizer(max_features=5000, ngram_range=(1, 2))
    X = fit_transform(vectorizer, df["clean_message"])
    y = df["label_num"].values
    print(f"      Feature matrix shape: {X.shape} (messages x features)")

    # 5. Split into training and testing sets (80% train, 20% test).
    print("\n[5/8] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 6. Train Multinomial Naive Bayes.
    print("\n[6/8] Training Multinomial Naive Bayes...")
    model = train_model(X_train, y_train, alpha=1.0)
    print("      Model trained.")

    # 7. Tune the spam decision threshold with cross-validation (train set only).
    print("\n[7/8] Tuning spam cutoff (5-fold cross-validation on train set)...")
    threshold = tune_threshold_cv(X_train, y_train, alpha=1.0, cv=5)
    print(f"      Chosen cutoff: P(spam) >= {threshold:.2f} -> Spam")

    # 8. Evaluate on the untouched test set and save everything.
    print("\n[8/8] Evaluating the model on the test set...")
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    report = format_metrics(metrics)
    print("\n" + report)

    save_model(model, vectorizer, MODELS_DIR, threshold=threshold)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    save_evaluation_plot(metrics, os.path.join(OUTPUT_DIR, "06_confusion_matrix.png"))

    print("\nDone! Artifacts saved in:")
    print(f"  - {OUTPUT_DIR}  (plots + metrics.txt)")
    print(f"  - {MODELS_DIR}  (saved model + vectorizer + threshold)")
    print("\nNext step: launch the dashboard with  streamlit run app.py")


if __name__ == "__main__":
    main()
