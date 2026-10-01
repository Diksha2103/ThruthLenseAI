"""
TruthLenseAI -- Fake News Detection
train_model.py

Dataset: FakeNewsNet GossipCop split
  real=1 -> REAL news from credible outlets (people.com, etonline.com, wikipedia)
  real=0 -> FAKE/exaggerated tabloid news (hollywoodlife.com, radaronline.com)
  This mapping was verified from actual source_domain distributions, not guessed.

Run: py train_model.py
"""

import os
import glob
import json
import warnings
import joblib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

DATA_DIR     = "Data"
MODEL_PATH   = os.path.join("models", "fake_news_model.joblib")
REPORT_DIR   = "reports"
METRICS_PATH = os.path.join(REPORT_DIR, "metrics.json")
CM_PATH      = os.path.join(REPORT_DIR, "confusion_matrix.png")

# Verified from source_domain analysis (not guessed):
# Label 1 sources: people.com, etonline.com, usmagazine.com  -> credible
# Label 0 sources: hollywoodlife.com, radaronline.com         -> tabloid/exaggerated
LABEL_MAP = {0: "FAKE", 1: "REAL"}


def find_csv(data_dir: str) -> str:
    csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {data_dir!r}")
    return csvs[0]


def load_and_inspect(path: str) -> pd.DataFrame:
    sep = "=" * 60
    print(f"\n{sep}\n  STEP 1 -- Loading and Inspecting\n{sep}")
    df = pd.read_csv(path)
    print(f"  File   : {path}")
    print(f"  Shape  : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Columns: {df.columns.tolist()}")
    print("\n  Missing values:")
    for col, n in df.isnull().sum().items():
        print(f"    {col:<20}: {n}")
    print(f"\n  Duplicate rows  : {df.duplicated().sum()}")
    print(f"  Duplicate titles: {df['title'].duplicated().sum()}")
    print("\n  Label distribution (real column):")
    vc = df["real"].value_counts()
    for lbl, cnt in vc.items():
        pct = cnt / len(df) * 100
        print(f"    {lbl} -> {cnt}  ({pct:.1f}%)")
    print("\n  Sample headlines per label:")
    for lbl in [0, 1]:
        print(f"\n  Label {lbl} (= {LABEL_MAP[lbl]}):")
        for t in df[df["real"] == lbl]["title"].dropna().head(4):
            print(f"    * {str(t)[:90]}")
    print("\n  Top source domains per label (label verification):")
    for lbl in [0, 1]:
        top = df[df["real"] == lbl]["source_domain"].value_counts().head(4).to_dict()
        print(f"    Label {lbl} ({LABEL_MAP[lbl]}): {top}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    sep = "=" * 60
    print(f"\n{sep}\n  STEP 2 -- Cleaning Data\n{sep}")
    before = len(df)
    df = df[["title", "real"]].copy()
    df.dropna(subset=["title", "real"], inplace=True)
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"] != ""]
    df = df[df["title"].str.len() > 5]
    df.drop_duplicates(subset=["title"], keep="first", inplace=True)
    df["real"] = df["real"].astype(int)
    print(f"  Before: {before}  After: {len(df)}  Removed: {before - len(df)}")
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    sep = "=" * 60
    print(f"\n{sep}\n  STEP 3 -- Train/Test Split (80/20, stratified)\n{sep}")
    X, y = df["title"], df["real"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}")
    return X_tr, X_te, y_tr, y_te


def build_pipelines() -> dict:
    """
    TF-IDF converts headlines into numerical feature vectors.
    ngram_range=(1,2): captures single words AND two-word phrases.
    sublinear_tf=True: uses log(TF) to reduce dominance of very frequent words.
    min_df=2: ignores words that appear only once (likely noise).
    TF-IDF is fit on TRAINING data only -- no data leakage.
    """
    tfidf_kwargs = dict(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
    )
    return {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf",   LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", random_state=42
            )),
        ]),
    }


def train_and_evaluate(pipelines, X_tr, X_te, y_tr, y_te):
    sep = "=" * 60
    print(f"\n{sep}\n  STEP 4 -- Training and Evaluating Models\n{sep}")
    results = {}
    for name, pipe in pipelines.items():
        print(f"\n  ---- {name} ----")
        cv = cross_val_score(
            pipe, X_tr, y_tr, cv=5, scoring="f1_weighted", n_jobs=-1
        )
        print(f"  CV F1 (5-fold): {cv.mean():.4f} +/- {cv.std():.4f}")
        pipe.fit(X_tr, y_tr)
        yp   = pipe.predict(X_te)
        acc  = accuracy_score(y_te, yp)
        prec = precision_score(y_te, yp, average="weighted")
        rec  = recall_score(y_te, yp, average="weighted")
        f1   = f1_score(y_te, yp, average="weighted")
        print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
        tnames = [LABEL_MAP[k] for k in sorted(LABEL_MAP)]
        print(classification_report(y_te, yp, target_names=tnames))
        results[name] = dict(
            pipeline=pipe, y_pred=yp,
            accuracy=round(acc, 4), precision=round(prec, 4),
            recall=round(rec, 4), f1=round(f1, 4),
            cv_mean=round(cv.mean(), 4), cv_std=round(cv.std(), 4),
        )
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\n  Best model: {best}  (F1 = {results[best]['f1']})")
    return best, results[best]["pipeline"], results


def save_confusion_matrix(y_te, y_pred, path: str):
    cm = confusion_matrix(y_te, y_pred)
    labels = [LABEL_MAP[k] for k in sorted(LABEL_MAP)]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels,
                cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix -> {path}")


def save_metrics(results, best_name, y_te, path: str):
    best   = results[best_name]
    y_pred = best["y_pred"]
    tnames = [LABEL_MAP[k] for k in sorted(LABEL_MAP)]
    report = classification_report(y_te, y_pred, target_names=tnames, output_dict=True)
    cm     = confusion_matrix(y_te, y_pred).tolist()
    comparison = {
        n: {k: v for k, v in r.items() if k not in ("pipeline", "y_pred")}
        for n, r in results.items()
    }
    payload = dict(
        best_model=best_name,
        label_map={str(k): v for k, v in LABEL_MAP.items()},
        accuracy=best["accuracy"],
        precision=best["precision"],
        recall=best["recall"],
        f1=best["f1"],
        cv_f1_mean=best["cv_mean"],
        cv_f1_std=best["cv_std"],
        confusion_matrix=cm,
        class_report=report,
        model_comparison=comparison,
        dataset_note=(
            "FakeNewsNet GossipCop split. "
            "Labels verified from source_domain distributions: "
            "1=REAL (people.com, etonline.com), 0=FAKE (hollywoodlife.com, radaronline.com). "
            "Headlines are celebrity news articles."
        ),
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Metrics -> {path}")


def sanity_check(model_path: str):
    sep = "=" * 60
    print(f"\n{sep}\n  STEP 6 -- Sanity Check (live predictions)\n{sep}")
    pipe = joblib.load(model_path)
    tests = [
        "Taylor Swift wins Grammy for Album of the Year",
        "SHOCKING: Taylor Swift secretly married alien lizard overlord",
        "Beyonce announces world tour alongside new album",
        "Hollywood pedophiles have nowhere left to hide according to insider",
    ]
    for h in tests:
        pred  = pipe.predict([h])[0]
        label = LABEL_MAP[pred]
        conf  = round(max(pipe.predict_proba([h])[0]) * 100, 1)
        print(f"  [{label:4s}  {conf:5.1f}%]  {h}")


def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  TruthLens AI -- Model Training")
    print(f"{sep}")

    csv_path = find_csv(DATA_DIR)
    df       = load_and_inspect(csv_path)
    df       = clean_data(df)
    X_tr, X_te, y_tr, y_te = split_data(df)
    pipes    = build_pipelines()
    best_name, best_pipe, results = train_and_evaluate(pipes, X_tr, X_te, y_tr, y_te)

    os.makedirs("models",   exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    sep2 = "=" * 60
    print(f"\n{sep2}\n  STEP 5 -- Saving Artifacts\n{sep2}")
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"  Model saved -> {MODEL_PATH}")
    save_confusion_matrix(y_te, results[best_name]["y_pred"], CM_PATH)
    save_metrics(results, best_name, y_te, METRICS_PATH)
    sanity_check(MODEL_PATH)

    print(f"\n{sep2}")
    print("  Training complete!")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Metrics: {METRICS_PATH}")
    print(f"{sep2}\n")


if __name__ == "__main__":
    main()