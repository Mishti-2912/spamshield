# 🛡️ SpamShield — Spam & Phishing Email Classifier using Naive Bayes

A machine-learning project that classifies messages/emails as **Spam** or
**Legitimate (ham)** using **TF-IDF** feature extraction and a **Multinomial
Naive Bayes** classifier. It includes a full training pipeline, exploratory
data analysis with charts, and an interactive **Streamlit** dashboard where
you can type any email and instantly see whether it is spam.

Built for a **Probability & Statistics** college project.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Tech Stack](#-tech-stack)
3. [Project Structure](#-project-structure)
4. [Setup & Installation](#-setup--installation)
5. [How to Run](#-how-to-run)
6. [How It Works (explained simply)](#-how-it-works-explained-simply)
7. [Evaluation Metrics Explained](#-evaluation-metrics-explained)
8. [The Math behind Naive Bayes](#-the-math-behind-naive-bayes)
9. [Sample Results](#-sample-results)
10. [Troubleshooting](#-troubleshooting)

---

## 🔍 Project Overview

Spam and phishing emails waste time, steal data, and cost people money. The
goal of this project is to build a model that can automatically tell a
spam/phishing message apart from a legitimate one.

The model is trained on the **SMS Spam Collection** dataset — a public dataset
of 5,500+ messages labelled as `ham` (legitimate) or `spam`. After training, it
can classify any new message with a confidence score.

What this project covers:
1. **Dataset collection & loading** — auto-downloads on first run.
2. **Data cleaning & preprocessing** — lowercase, remove links/symbols/stop words.
3. **Exploratory Data Analysis (EDA)** — class balance, lengths, top words.
4. **Data visualization** — matplotlib & seaborn charts.
5. **TF-IDF feature extraction** — turning words into numbers.
6. **Multinomial Naive Bayes** — the probability-based classifier.
7. **Model evaluation** — accuracy, precision, recall, F1, confusion matrix.
8. **Interactive Streamlit dashboard** — explore the data and model live.
9. **Live prediction** — type any email and get a Spam / Legitimate verdict.
10. **Clean project structure** — modular, well-commented, beginner-friendly.

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Programming language |
| **pandas** | Loading & handling the dataset (DataFrames) |
| **NumPy** | Numerical operations under the hood |
| **scikit-learn** | TF-IDF, Naive Bayes, train/test split, metrics |
| **matplotlib** | Drawing charts |
| **seaborn** | Nicer statistical charts |
| **Streamlit** | Interactive web dashboard |
| **joblib** | Saving & loading the trained model |

---

## 📁 Project Structure

```
SpamShield/
├── README.md                 # This file
├── requirements.txt           # Python dependencies
├── .gitignore
├── main.py                    # Full training pipeline (run once)
├── app.py                     # Streamlit interactive dashboard
├── analyze_threshold.py      # Inspect the precision-recall tradeoff
├── data/
│   ├── download_dataset.py    # Script to download the dataset
│   └── spam.csv               # The dataset (auto-downloaded on first run)
├── src/                       # Core source code (the "engine")
│   ├── __init__.py
│   ├── data_loader.py         # 1. Load / download the dataset
│   ├── preprocessing.py        # 2. Clean raw text
│   ├── eda.py                  # 3 & 4. EDA + visualizations
│   ├── features.py             # 5. TF-IDF feature extraction
│   ├── model.py                # 6. Multinomial Naive Bayes
│   ├── evaluate.py             # 7. Evaluation metrics
│   └── predict.py             # 9. Classify a single message
├── models/                    # Saved model + vectorizer (generated)
└── outputs/                   # Saved plots + metrics (generated)
```

---

## ⚙️ Setup & Installation

### Step 1 — Install Python
Make sure you have **Python 3.10 or newer**. Check with:
```bash
python --version
```

### Step 2 — (Recommended) create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install the dependencies
From the project root folder run:
```bash
pip install -r requirements.txt
```

That installs pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib.

---

## ▶️ How to Run

### Option A — Train the model from the command line (recommended first)
This downloads the dataset, runs EDA, trains the model, evaluates it, and saves
all plots + the saved model:
```bash
python main.py
```
After it finishes you will find:
- Charts in the `outputs/` folder.
- A text report in `outputs/metrics.txt`.
- The saved model in `models/`.

### Option B — Launch the interactive dashboard
```bash
streamlit run app.py
```
Your browser will open with the dashboard. Use the sidebar to move between:
- **Overview** — project description + dataset statistics.
- **EDA** — interactive data-analysis charts.
- **Model Performance** — accuracy, precision, recall, F1, confusion matrix.
- **Live Prediction** — type an email and get a Spam / Legitimate verdict.

> The dashboard trains the model automatically on first launch (cached, so it
> only happens once) and works even if you never ran `main.py`.

### Option C — Inspect the precision-recall tradeoff
To see how precision and recall change as you lower the spam cutoff, run:
```bash
python analyze_threshold.py
```
This uses cross-validation on the training set (the test set stays untouched)
and prints the tradeoff table plus the final test-set performance.

---

## 🧠 How It Works (explained simply)

### 1. Loading the data
We read a CSV file into a pandas **DataFrame** (a table). Each row is a message
with two columns: `label` (`ham`/`spam`) and `message` (the text). If the file is
missing, it downloads automatically.

### 2. Cleaning the text
Raw text is messy. We clean each message by:
- Lowercasing everything.
- Removing HTML tags, web links, and email addresses.
- Removing symbols (keeping only letters and numbers, because prize amounts
  and codes like "1000" are useful spam signals).
- Removing **stop words** — common words like "the", "and", "is" that appear
  in both spam and ham and so don't help tell them apart.

### 3. Exploratory Data Analysis (EDA)
Before training, we look at the data:
- How many spam vs ham messages are there?
- How long are spam messages compared to ham ones?
- Which words appear most often in spam vs ham?

### 4. TF-IDF (turning words into numbers)
Models only understand numbers. **TF-IDF** gives each word a score:
- **TF (Term Frequency)** — how often a word appears in a message.
- **IDF (Inverse Document Frequency)** — how rare the word is overall.

A word like "winner" that appears often in spam but rarely in ham gets a high
score — exactly the signal the model needs.

### 5. Train / Test Split
We split the data 80% (train) / 20% (test). The model learns from the training
set and is then tested on data it has never seen, which tells us how it will
perform in the real world.

### 6. Multinomial Naive Bayes
This classifier is built for word-count/text features. Using Bayes' theorem
(from probability theory), it estimates the probability that a message is spam
given its words, then picks the more likely label. It is fast, simple, and very
effective for spam detection.

### 7. Tune the spam cutoff & evaluate
By default a message is called Spam when P(spam) > 0.5. We use 5-fold
cross-validation on the *training* set to find a better cutoff that catches more
spam, then report accuracy, precision, recall, F1, and a confusion matrix on
the untouched *test* set (see the precision-recall tradeoff below).

### 8 & 9. Dashboard & live prediction
The Streamlit app lets you explore everything interactively and type your own
message to get an instant Spam / Legitimate verdict with a confidence score.

---

## 📊 Evaluation Metrics Explained

| Metric | Question it answers | Formula |
|--------|--------------------|---------|
| **Accuracy** | Overall, how many did we get right? | (correct) / (total) |
| **Precision (spam)** | Of messages we flagged as spam, how many really were? | TP / (TP + FP) |
| **Recall (spam)** | Of all real spam, how much did we catch? | TP / (TP + FN) |
| **F1-score** | Balance of precision and recall | 2·(P·R) / (P + R) |

Where:
- **TP** (True Positive) = spam correctly called spam.
- **FP** (False Positive) = ham wrongly called spam (a false alarm).
- **FN** (False Negative) = spam wrongly called ham (missed spam).
- **TN** (True Negative) = ham correctly called ham.

The **confusion matrix** shows all four counts in one 2×2 table.

---

## ➗ The Math behind Naive Bayes

Naive Bayes uses **Bayes' theorem**:

```
P(spam | message) = P(message | spam) * P(spam) / P(message)
```

Because `P(message)` is the same for both classes, we can compare just:

```
score(spam) = P(spam) * P(word1 | spam) * P(word2 | spam) * ... 
score(ham)  = P(ham)  * P(word1 | ham)  * P(word2 | ham)  * ...
```

- `P(spam)` and `P(ham)` are the **prior** probabilities (the chance a random
  message is spam/ham, based on class counts in the dataset).
- `P(word | spam)` is the probability of seeing that word in spam messages.
- "Naive" means we assume each word is independent of the others — a
  simplification that works well in practice.

**Laplace smoothing (alpha = 1):** if a word never appeared in spam during
training, its probability would be 0 and make the whole product 0. Smoothing
adds a small count to every word so probabilities never collapse to zero.

---

## ⚖️ The Precision-Recall Tradeoff & Threshold Tuning

A classifier normally calls a message "Spam" when its probability P(spam) is
above **0.5**. But for a spam/phishing filter, *missing* spam (a false negative)
is usually worse than a few false alarms (false positives). So we can lower
that cutoff to catch more spam — at the cost of slightly lower precision.

This is the **precision-recall tradeoff**: you cannot maximize both at once, so
you pick a balance that fits the use case.

**How we choose the cutoff in this project:**
1. Split the data into a **training set** (80%) and a **test set** (20%).
2. On the training set only, use **5-fold cross-validation** to get honest
   spam-probability estimates without touching the test set.
3. Try many cutoffs (0.05 → 0.50) and keep the one with the best **F1 score**
   (the balance of precision and recall).
4. Report the final metrics on the **untouched test set** using that cutoff.

Lowering the cutoff from the default 0.5 to the tuned value (~0.23) raised
**recall from ~80% to ~93%** (we catch far more spam) while keeping **precision
above 95%** (very few false alarms). The cutoff is saved in
`models/spamshield_meta.json` and reused by the live prediction feature.

---

## 📈 Sample Results

Running `python main.py` typically gives results close to:

```
Spam cutoff : ~0.23  (tuned with 5-fold CV on the training set)
Accuracy  : ~0.984  (98.4%)
Precision : ~0.95
Recall    : ~0.93   (catches ~93% of spam)
F1-score  : ~0.94
```

Exact numbers depend on the dataset version and scikit-learn version. The
dashboard shows your actual results live.

---

## 🛠️ Troubleshooting

**"Could not download the dataset"**
Check your internet connection and rerun. If you are offline, manually download
the SMS Spam Collection dataset, rename it to `spam.csv` with two columns
`label` (ham/spam) and `message`, and place it in the `data/` folder.

**`streamlit` command not found**
Make sure you activated your virtual environment and ran
`pip install -r requirements.txt`. You can also launch it with
`python -m streamlit run app.py`.

**Encoding errors when reading the CSV**
The loader already tries `utf-8` then `latin-1` automatically. If you supply your
own CSV, keep it UTF-8 encoded.

**Port already in use (Streamlit)**
Run `streamlit run app.py --server.port 8502` to use a different port.

---

## 📚 Dataset Credits

SMS Spam Collection Dataset — public dataset widely used for spam-detection
research. Downloaded automatically from a public mirror on first run.

---

## 👤 Author / Submission

**Project:** SpamShield — Spam & Phishing Email Classifier using Naive Bayes
**Subject:** Probability & Statistics
**Algorithm:** Multinomial Naive Bayes + TF-IDF
