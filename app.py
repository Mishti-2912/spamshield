"""SpamShield - Interactive Streamlit Dashboard.

Run it from the project root with:
    streamlit run app.py

The dashboard has four sections (select from the sidebar):
  - Overview          : project description and dataset statistics
  - EDA               : exploratory data-analysis charts
  - Model Performance : accuracy, precision, recall, F1, confusion matrix
  - Live Prediction   : type an email and check if it is Spam or Legitimate
"""

import os

import streamlit as st

from src.data_loader import add_numeric_label, load_data
from src.eda import (
    dataset_summary,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_message_length_distribution,
    plot_top_words,
    plot_word_count_distribution,
)
from src.evaluate import evaluate_model
from src.features import build_vectorizer, fit_transform
from src.model import save_model, train_model, tune_threshold_cv
from src.predict import predict_message
from src.preprocessing import apply_cleaning
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------ Page setup
st.set_page_config(
    page_title="SpamShield - Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for a clean, professional look.
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; max-width: 1100px; }
    h1, h2, h3 { color: #1f2d3d; }
    .big-title { font-size: 2.1rem; font-weight: 800; color: #1f3a5f; }
    .subtitle { font-size: 1.05rem; color: #52606d; }
    .spam-bad  { color: #e55039; font-weight: 700; }
    .ham-good  { color: #2e86de; font-weight: 700; }
    [data-testid="stMetric"] {
        background-color: #f7f9fc;
        border: 1px solid #e3e8ef;
        border-radius: 10px;
        padding: 14px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ------------------------------------------------------------------ Caching
@st.cache_data(show_spinner="Loading and cleaning the dataset...")
def get_data():
    """Load, clean, and cache the dataset (runs once)."""
    df = load_data()
    df = add_numeric_label(df)
    df = apply_cleaning(df)
    return df


@st.cache_resource(show_spinner="Training the Naive Bayes model (first run only)...")
def get_trained_model():
    """Train the model once and cache it for the whole session."""
    df = get_data()
    vectorizer = build_vectorizer(max_features=5000, ngram_range=(1, 2))
    X = fit_transform(vectorizer, df["clean_message"])
    y = df["label_num"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = train_model(X_train, y_train, alpha=1.0)
    # Pick the best spam cutoff using cross-validation on the training set only,
    # so the test set stays a fair measure of performance.
    threshold = tune_threshold_cv(X_train, y_train, alpha=1.0, cv=5)
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    # Persist the trained model so saved artifacts match this run (best effort).
    try:
        save_model(model, vectorizer, MODELS_DIR, threshold=threshold)
    except Exception:
        pass
    return model, vectorizer, threshold, metrics


# ------------------------------------------------------------------ Sidebar
st.sidebar.markdown("## 🛡️ SpamShield")
st.sidebar.markdown("**Spam & Phishing Email Classifier**")
st.sidebar.caption("Naive Bayes • TF-IDF • Streamlit")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 EDA", "📈 Model Performance", "✉️ Live Prediction"],
)
st.sidebar.divider()
st.sidebar.markdown("Probability & Statistics Project")


# ------------------------------------------------------------------ Overview
def show_overview():
    st.markdown('<p class="big-title">SpamShield</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">A Spam &amp; Phishing Email Classifier powered by '
        "Multinomial Naive Bayes and TF-IDF.</p>",
        unsafe_allow_html=True,
    )
    st.write("---")

    st.markdown("### What does this project do?")
    st.write(
        "SpamShield reads a dataset of SMS/email messages already labelled as "
        "**spam** (junk / phishing) or **ham** (legitimate). It learns the patterns "
        "of each group using **TF-IDF** (to turn words into numbers) and "
        "**Multinomial Naive Bayes** (a probability-based classifier based on "
        "Bayes' theorem). You can then type your own message and the app tells "
        "you whether it is Spam or Legitimate, along with a confidence score."
    )

    st.markdown("### How the pipeline works")
    steps = [
        ("1. Load data", "Read the spam/ham dataset into a pandas DataFrame."),
        ("2. Clean text", "Lowercase, remove links, symbols, and stop words."),
        ("3. EDA", "Explore class balance, message lengths, and top words."),
        ("4. TF-IDF", "Convert cleaned text into numeric features."),
        ("5. Train NB", "Train Multinomial Naive Bayes on the features."),
        ("6. Evaluate", "Measure accuracy, precision, recall, and F1."),
        ("7. Predict", "Classify any new message in real time."),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")

    st.write("---")
    st.markdown("### Dataset at a glance")
    df = get_data()
    summary = dataset_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total messages", f"{summary['total_messages']:,}")
    c2.metric("Spam", f"{summary['spam_count']:,}", f"{summary['spam_percent']}%")
    c3.metric("Legitimate", f"{summary['ham_count']:,}", f"{summary['ham_percent']}%")
    c4.metric("Avg words/msg", f"{summary['avg_words']}")

    st.markdown("#### Sample rows from the dataset")
    st.dataframe(df[["label", "message"]].head(8), width="stretch", height=240)
    st.caption("label = 'ham' means legitimate; 'spam' means junk / phishing.")


# ------------------------------------------------------------------ EDA
def show_eda():
    st.markdown('<p class="big-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.write("Understanding the data before training the model.")
    st.write("---")
    df = get_data()

    st.markdown("#### 1. Class distribution")
    st.write("How many spam vs legitimate messages are in the dataset?")
    st.pyplot(plot_class_distribution(df))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 2. Message length")
        st.write("Distribution of character counts for each class.")
        st.pyplot(plot_message_length_distribution(df))
    with c2:
        st.markdown("#### 3. Word count")
        st.write("Distribution of word counts for each class.")
        st.pyplot(plot_word_count_distribution(df))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 4. Top words in Spam")
        st.write("Most frequent words (after cleaning) in spam messages.")
        st.pyplot(plot_top_words(df, label="spam"))
    with c2:
        st.markdown("#### 5. Top words in Legitimate")
        st.write("Most frequent words (after cleaning) in legitimate messages.")
        st.pyplot(plot_top_words(df, label="ham"))


# ------------------------------------------------------------------ Model perf
def show_model_performance():
    st.markdown('<p class="big-title">Model Performance</p>', unsafe_allow_html=True)
    st.write("How well does the Multinomial Naive Bayes model classify messages?")
    st.write("---")

    with st.spinner("Training model if needed..."):
        model, vectorizer, threshold, metrics = get_trained_model()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c2.metric("Precision (spam)", f"{metrics['precision']*100:.2f}%")
    c3.metric("Recall (spam)", f"{metrics['recall']*100:.2f}%")
    c4.metric("F1-score (spam)", f"{metrics['f1']*100:.2f}%")

    st.caption(f"Spam decision cutoff: P(spam) >= {metrics['threshold']:.2f} "
               f"(tuned with 5-fold cross-validation on the training set to "
               f"balance catching spam vs. false alarms).")

    st.write("---")
    st.markdown("#### Confusion Matrix")
    st.write("Rows = true label, columns = predicted label.")
    st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]))

    st.write("---")
    st.markdown("#### What do these metrics mean?")
    st.markdown(
        """
        - **Accuracy** — overall fraction of messages classified correctly.
        - **Precision (spam)** — of all messages we flagged as spam, the share that really were spam. High = few false alarms.
        - **Recall (spam)** — of all real spam messages, the share we successfully caught. High = few missed spam.
        - **F1-score** — the harmonic mean of precision and recall; a single balance score, useful when classes are uneven.
        """
    )

    st.write("---")
    st.markdown("#### Full classification report")
    st.code(metrics["classification_report"], language="text")


# ------------------------------------------------------------------ Live pred
def show_live_prediction():
    st.markdown('<p class="big-title">Live Prediction</p>', unsafe_allow_html=True)
    st.write("Type or paste an email/message below to check whether it is "
             "**Spam** or **Legitimate**.")
    st.write("---")

    with st.spinner("Loading model..."):
        model, vectorizer, threshold, _ = get_trained_model()

    # Example buttons fill the text box with a sample message.
    st.markdown("#### Try an example")
    sample_spam = (
        "Congratulations! You have won a $1,000 Walmart gift card. "
        "Click here to claim your prize now: http://bit.ly/win-now"
    )
    sample_ham = (
        "Hey, are we still meeting for lunch at 1pm tomorrow? "
        "Let me know if the time works for you."
    )
    e1, e2 = st.columns(2)
    if e1.button("📨 Load spam example"):
        st.session_state["email_input"] = sample_spam
    if e2.button("✉️ Load legitimate example"):
        st.session_state["email_input"] = sample_ham

    st.text_area(
        "Enter your email/message:",
        key="email_input",
        height=160,
        placeholder="Paste a message here...",
    )

    if st.button("🔍 Classify message", type="primary", width="stretch"):
        user_input = st.session_state.get("email_input", "")
        if not user_input.strip():
            st.warning("Please enter a message to classify.")
            return
        result = predict_message(model, vectorizer, user_input, threshold=threshold)

        if result["label"] == 1:
            st.markdown("### Result: <span class='spam-bad'>🚫 SPAM</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("### Result: <span class='ham-good'>✅ LEGITIMATE</span>",
                        unsafe_allow_html=True)

        spam_pct = result["spam_prob"] * 100
        ham_pct = result["ham_prob"] * 100
        st.write("**Confidence:**")
        st.progress(float(result["spam_prob"]), text=f"Spam probability: {spam_pct:.1f}%")

        c1, c2 = st.columns(2)
        c1.metric("Spam probability", f"{spam_pct:.1f}%")
        c2.metric("Legitimate probability", f"{ham_pct:.1f}%")

        with st.expander("See the cleaned text the model used"):
            st.write(result["clean_message"])


# ------------------------------------------------------------------ Router
if page == "🏠 Overview":
    show_overview()
elif page == "📊 EDA":
    show_eda()
elif page == "📈 Model Performance":
    show_model_performance()
elif page == "✉️ Live Prediction":
    show_live_prediction()
