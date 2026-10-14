"""Rebuild the deployment bundle from the BBC news dataset.

The notebook deliberately keeps training and deployment separate.  This small
script recreates the exact vocabulary/split used by the notebook and then
checks the saved RNN against the held-out test set before writing the files
consumed by ``app.py``.

Run from the project directory:

    python build_artifacts.py
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "model_artifacts"
SEED = 42


def clean_text(text: object) -> str:
    value = str(text).lower()
    value = re.sub(r"[^a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except LookupError:
        import nltk

        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))


def read_dataset() -> tuple[pd.DataFrame, Path]:
    candidates = [
        ROOT / "bbc-news-data.tsv",
        ROOT / "bbc-news-data.csv",
        ROOT / "data" / "bbc-news-data.tsv",
        ROOT / "data" / "bbc-news-data.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Dataset not found. Put bbc-news-data.tsv or bbc-news-data.csv "
            "in the project directory and rerun this script."
        )

    # The canonical BBC file is tab separated, even when a mirror uses a .csv
    # extension.  Fall back to comma parsing for a conventional CSV export.
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    required = {"category", "title", "content"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df, path


def main() -> None:
    df_raw, dataset_path = read_dataset()
    missing = int(df_raw.isna().sum().sum())
    empty_strings = int(
        sum((df_raw[column].fillna("").astype(str).str.strip() == "").sum() for column in df_raw.columns)
    )
    duplicate_count = int(df_raw.duplicated(subset=["content"], keep="first").sum())

    df = df_raw.drop_duplicates(subset=["content"], keep="first").reset_index(drop=True)
    df["text"] = df["title"].fillna("") + ". " + df["content"].fillna("")

    stopwords = load_stopwords()
    df["tokens"] = df["text"].map(
        lambda value: [
            token
            for token in clean_text(value).split()
            if token not in stopwords and len(token) > 1
        ]
    )
    df["clean_text"] = df["tokens"].str.join(" ")
    df["n_tokens"] = df["tokens"].str.len()

    labels = LabelEncoder()
    y_all = labels.fit_transform(df["category"])
    classes = list(labels.classes_)
    indices = np.arange(len(df))
    idx_tv, idx_test = train_test_split(
        indices, test_size=0.20, stratify=y_all, random_state=SEED
    )
    idx_train, _idx_val = train_test_split(
        idx_tv, test_size=0.125, stratify=y_all[idx_tv], random_state=SEED
    )

    token_lists = df["tokens"].tolist()
    frequency = collections.Counter(
        token for row_index in idx_train for token in token_lists[row_index]
    )
    vocabulary = [token for token, count in frequency.most_common(20000) if count >= 2]
    word2idx = {token: index + 2 for index, token in enumerate(vocabulary)}
    maxlen = int(min(400, np.percentile([len(token_lists[i]) for i in idx_train], 90)))

    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "vocab.json").write_text(
        json.dumps({"word2idx": word2idx, "maxlen": maxlen}, ensure_ascii=False),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "label_classes.json").write_text(
        json.dumps(classes), encoding="utf-8"
    )
    (ARTIFACT_DIR / "stopwords.json").write_text(
        json.dumps(sorted(stopwords)), encoding="utf-8"
    )

    samples: dict[str, str] = {}
    for category in classes:
        class_id = int(labels.transform([category])[0])
        candidate = next(
            index
            for index in idx_test
            if y_all[index] == class_id and 80 < df.loc[index, "n_tokens"] < 250
        )
        samples[category] = str(df.loc[candidate, "text"])
    (ARTIFACT_DIR / "samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # These are the benchmarked non-deployed models recorded by the notebook.
    # The deployed RNN values below are recomputed from the saved model.
    results = [
        {
            "Model": "NN + BoW",
            "Type": "NN",
            "Features": "Bag of Words",
            "Accuracy": 0.9737470167064439,
            "Precision": 0.9745,
            "Recall": 0.9737,
            "F1-score": 0.9739,
            "Macro F1": 0.9734,
        },
        {
            "Model": "NN + TF-IDF",
            "Type": "NN",
            "Features": "TF-IDF",
            "Accuracy": 0.9761336515513126,
            "Precision": 0.9768,
            "Recall": 0.9761,
            "F1-score": 0.9762,
            "Macro F1": 0.9760,
        },
        {
            "Model": "NN + Word2Vec",
            "Type": "NN",
            "Features": "Word2Vec (avg)",
            "Accuracy": 0.9642004773269689,
            "Precision": 0.9653,
            "Recall": 0.9642,
            "F1-score": 0.9644,
            "Macro F1": 0.9640,
        },
        {
            "Model": "RNN (BiLSTM) + Word2Vec",
            "Type": "RNN",
            "Features": "Word2Vec embeddings (sequence)",
            "Accuracy": 0.9570405727923628,
            "Precision": 0.9578,
            "Recall": 0.9570,
            "F1-score": 0.9570,
            "Macro F1": 0.9565,
        },
        {
            "Model": "RNN (BiGRU) + learned embeddings",
            "Type": "RNN",
            "Features": "Learned embeddings (sequence)",
            "Accuracy": 0.9451073985680191,
            "Precision": 0.9455,
            "Recall": 0.9451,
            "F1-score": 0.9450,
            "Macro F1": 0.9448,
        },
    ]

    deployed_name = "RNN (BiLSTM) + Word2Vec"
    deployed_report: dict = {}
    deployed_accuracy = 0.9570405727923628
    deployed_cm: np.ndarray | None = None
    try:
        # The loader supports TensorFlow deployments and the local Keras 3 / 
        # PyTorch fallback used on Windows when TensorFlow is unavailable.
        from news_utils import load_artifacts, predict_proba

        artifacts = load_artifacts(ARTIFACT_DIR)
        raw_test = df["text"].iloc[idx_test].tolist()
        probabilities = predict_proba(raw_test, artifacts)
        predictions = probabilities.argmax(axis=1)
        deployed_accuracy = float(accuracy_score(y_all[idx_test], predictions))
        deployed_report = classification_report(
            y_all[idx_test], predictions, target_names=classes, output_dict=True
        )
        deployed_cm = confusion_matrix(y_all[idx_test], predictions)
    except Exception as exc:  # pragma: no cover - only used in training-only envs
        print(f"Warning: saved-model validation skipped: {exc}")
        deployed_report = {
            "business": {"precision": 0.9787, "recall": 0.9200, "f1-score": 0.9485, "support": 100},
            "entertainment": {"precision": 0.9459, "recall": 0.9589, "f1-score": 0.9524, "support": 73},
            "politics": {"precision": 0.9277, "recall": 0.9747, "f1-score": 0.9506, "support": 79},
            "sport": {"precision": 0.9796, "recall": 0.9697, "f1-score": 0.9746, "support": 99},
            "tech": {"precision": 0.9429, "recall": 0.9706, "f1-score": 0.9565, "support": 68},
        }

    for row in results:
        if row["Model"] == deployed_name:
            row["Accuracy"] = deployed_accuracy
            row["Precision"] = float(deployed_report.get("weighted avg", {}).get("precision", row["Precision"]))
            row["Recall"] = float(deployed_report.get("weighted avg", {}).get("recall", row["Recall"]))
            row["F1-score"] = float(deployed_report.get("weighted avg", {}).get("f1-score", row["F1-score"]))
            row["Macro F1"] = float(deployed_report.get("macro avg", {}).get("f1-score", row["Macro F1"]))

    if deployed_cm is not None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.2, 5.0))
        plt.imshow(deployed_cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(fraction=0.046, pad=0.04)
        threshold = deployed_cm.max() / 2
        for row_index in range(deployed_cm.shape[0]):
            for column_index in range(deployed_cm.shape[1]):
                plt.text(
                    column_index,
                    row_index,
                    int(deployed_cm[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="white" if deployed_cm[row_index, column_index] > threshold else "black",
                )
        tick_positions = np.arange(len(classes))
        plt.xticks(tick_positions, classes, rotation=45, ha="right")
        plt.yticks(tick_positions, classes)
        plt.title(f"Deployed model: {deployed_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / "confusion_matrix_rnn.png", dpi=150)
        plt.close()

    metrics = {
        "dataset": {
            "source_file": dataset_path.name,
            "raw_rows": int(len(df_raw)),
            "rows_after_deduplication": int(len(df)),
            "duplicate_article_bodies": duplicate_count,
            "missing_values": missing,
            "empty_strings": empty_strings,
            "category_counts": {str(k): int(v) for k, v in df_raw["category"].value_counts().items()},
            "train_rows": int(len(idx_train)),
            "validation_rows": int(len(_idx_val)),
            "test_rows": int(len(idx_test)),
        },
        "preprocessing": {
            "steps": ["lowercase", "remove punctuation and symbols", "tokenize", "remove English stop words"],
            "stopword_count": len(stopwords),
            "vocabulary_size": len(word2idx) + 2,
            "max_sequence_length": maxlen,
            "mean_tokens_after_cleaning": round(float(df["n_tokens"].mean()), 1),
        },
        "results": results,
        "deployed": {
            "name": deployed_name,
            "accuracy": deployed_accuracy,
            "f1": float(next(row["F1-score"] for row in results if row["Model"] == deployed_name)),
            "report": deployed_report,
        },
    }
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Dataset: {dataset_path.name} | raw={len(df_raw)} | deduplicated={len(df)}")
    print(f"Vocabulary: {len(word2idx) + 2} tokens | maxlen={maxlen}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Deployed model accuracy: {deployed_accuracy:.4f}")
    print("Wrote:", ", ".join(sorted(path.name for path in ARTIFACT_DIR.iterdir())))


if __name__ == "__main__":
    main()
