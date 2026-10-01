# TruthLens AI — Fake News Detection

A beginner-friendly NLP + Machine Learning project that detects potentially fake or exaggerated news headlines using TF-IDF feature extraction and classic ML classifiers.

---

## Project Overview

TruthLens AI analyses the **text pattern** of a news headline and predicts whether it is likely **REAL** or **FAKE**, based on linguistic patterns learned from thousands of training examples.

> **Important:** This model detects sensationalist language patterns — it does NOT fact-check specific claims or verify sources.

---

## Dataset

**FakeNewsNet — GossipCop Split**

| Property | Value |
|---|---|
| File | `Data/Fake_News_Net.csv` |
| Total records | 23,196 |
| Columns | title, news_url, source_domain, tweet_num, real |
| Feature used | `title` (headline text only) |
| Target column | `real` (0 or 1) |

### Label Verification (not guessed)

The meaning of labels was confirmed by inspecting actual `source_domain` distributions:

| Label | Meaning | Top sources |
|---|---|---|
| 1 | **REAL** | people.com, etonline.com, usmagazine.com, en.wikipedia.org |
| 0 | **FAKE** | hollywoodlife.com, radaronline.com (tabloid/exaggerated) |

Metadata columns (`news_url`, `source_domain`, `tweet_num`) were **excluded** from model features to prevent shortcuts.

---

## Technologies

- Python 3.x
- pandas, NumPy
- scikit-learn (TF-IDF, Logistic Regression, Multinomial Naive Bayes, Pipeline)
- joblib (model persistence)
- matplotlib, seaborn (charts)
- Streamlit (web UI)

---

## Project Structure

```
Fake_News/
├── app.py                  # Streamlit UI
├── train_model.py          # Model training script
├── requirements.txt
├── README.md
├── .gitignore
├── Data/
│   └── Fake_News_Net.csv   # Original dataset (do not modify)
├── models/
│   └── fake_news_model.joblib  # Saved trained pipeline
└── reports/
    ├── metrics.json            # Actual evaluation results
    └── confusion_matrix.png    # Confusion matrix chart
```

---

## Setup Instructions (Windows)

### 1. Make sure Python is installed

```powershell
py --version
```

### 2. (Optional) Create a virtual environment

```powershell
py -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
py -m pip install -r requirements.txt
```

---

## Training the Model

```powershell
py train_model.py
```

This will:
1. Load and inspect the dataset
2. Clean the data (remove duplicates, blank titles)
3. Split 80% train / 20% test (stratified)
4. Train and cross-validate Multinomial Naive Bayes and Logistic Regression
5. Evaluate on the held-out test set
6. Save the best model to `models/fake_news_model.joblib`
7. Save metrics to `reports/metrics.json`
8. Save confusion matrix to `reports/confusion_matrix.png`

---

## Launching the App

```powershell
py -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Actual Evaluation Results

> These are real numbers from the training run — never invented.

**Best Model: Multinomial Naive Bayes**

| Metric | Value |
|---|---|
| Test Accuracy | 83.80% |
| Weighted Precision | 82.90% |
| Weighted Recall | 83.80% |
| Weighted F1-Score | 82.74% |
| CV F1 (5-fold) | 82.70% ± 0.49% |

**Model Comparison:**

| Model | Accuracy | F1 (weighted) | CV F1 |
|---|---|---|---|
| Multinomial Naive Bayes | 83.80% | 82.74% | 82.70% |
| Logistic Regression | 83.75% | 81.62% | 81.39% |

### Per-Class Metrics (Test Set)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| FAKE | 74% | 52% | 61% |
| REAL | 86% | 94% | 90% |

> FAKE detection is lower due to class imbalance (25% FAKE vs 75% REAL).

---

## Limitations

1. **Headlines only** — the model only sees the title, not the full article.
2. **Celebrity gossip domain** — trained on GossipCop data; may not generalise to political or scientific misinformation.
3. **Language patterns, not facts** — detects writing style, not factual errors.
4. **Class imbalance** — FAKE class is underrepresented (25%); its recall is lower.
5. **No real-time verification** — no external APIs, web scraping, or databases used.

---

## Future Scope

- Include full article text for richer features
- Add PolitiFact political news dataset
- Use BERT or RoBERTa for contextual embeddings
- Integrate LIME/SHAP for prediction explainability
- Deploy as a REST API with FastAPI

---

## HCL Project Review — Likely Questions & Answers

**Q1: Why did you choose TF-IDF instead of word embeddings?**
TF-IDF is simple, fast, interpretable, and works well for short text like headlines. It requires no external pre-trained models, making the project beginner-friendly and self-contained.

**Q2: How did you confirm what 0 and 1 mean in the `real` column?**
We inspected the actual `source_domain` column for each label. Label 1 headlines come from people.com, wikipedia, etonline.com (credible). Label 0 headlines come from hollywoodlife.com, radaronline.com (tabloid/exaggerated). This is consistent with FakeNewsNet documentation.

**Q3: How did you prevent data leakage?**
The TF-IDF vectorizer is wrapped inside a scikit-learn Pipeline and fitted only on the training set. The test set is transformed (not fitted) using the already-trained vectorizer, so test data never influences the vocabulary or IDF weights.

**Q4: The FAKE class recall is only 52%. How would you improve it?**
Options include: (1) oversampling FAKE examples with SMOTE, (2) using class_weight='balanced' in Logistic Regression, (3) collecting more FAKE examples, or (4) tuning the classification threshold.

**Q5: What are the limitations of this model in real-world use?**
The model can only analyse headline text patterns — it cannot verify facts, check sources, read full articles, or detect context-specific misinformation. It should be used as a supporting tool, not a definitive fact-checker.