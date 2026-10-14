from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np


class ArtifactError(RuntimeError):
    """Raised when the trained model bundle is incomplete or unreadable."""


def clean_text(text: object) -> str:
    """Lowercase text, remove punctuation/digits/symbols, and collapse spaces."""

    value = str(text).lower()
    value = re.sub(r"[^a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: object) -> list[str]:
    """Tokenize already-cleaned text on whitespace."""

    return clean_text(text).split()


def preprocess(text: object, stopwords: set[str]) -> list[str]:
    """Apply cleaning, tokenization, stop-word removal, and short-token removal."""

    return [token for token in tokenize(text) if token not in stopwords and len(token) > 1]


def encode(tokens: Iterable[str], word2idx: dict[str, int], maxlen: int) -> list[int]:
    """Map tokens to IDs; 0 is padding and 1 is out-of-vocabulary."""

    ids = [word2idx.get(token, 1) for token in tokens][:maxlen]
    return ids + [0] * (maxlen - len(ids))


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"Missing deployment artifact: {path.name}") from exc


def _load_keras():
    """Load Keras with TensorFlow when available, or the Windows fallback."""

    try:
        from tensorflow import keras

        return keras
    except ImportError:
        # TensorFlow is not distributed as a native Windows wheel for current
        # releases. Keras 3 can load and run this model through PyTorch.
        os.environ.setdefault("KERAS_BACKEND", "torch")
        try:
            import keras
        except ImportError as exc:
            raise ArtifactError(
                "A Keras backend is required. Install requirements.txt."
            ) from exc
        return keras


def load_artifacts(artifact_dir: str | Path) -> dict:
    """Load model metadata and the Keras model from an artifact directory.

    The inference-only ``rnn_model.keras`` archive omits optimizer state and is
    stored with ZIP compression to reduce its size. Loading with ``compile=False``
    is appropriate because the dashboard only runs predictions. The loader accepts
    the model beside the artifact directory or at the project root.
    """

    artifact_dir = Path(artifact_dir)
    # Prefer the compact model at the project root. If it is absent, allow the
    # notebook-generated copy inside model_artifacts/ as a fallback.
    model_path = artifact_dir.parent / "rnn_model.keras"
    if not model_path.exists():
        model_path = artifact_dir / "rnn_model.keras"
    if not model_path.exists():
        raise ArtifactError("Missing rnn_model.keras beside the artifact directory")

    keras = _load_keras()

    cfg = _read_json(artifact_dir / "vocab.json")
    classes = _read_json(artifact_dir / "label_classes.json")
    stopwords = _read_json(artifact_dir / "stopwords.json")
    model = keras.models.load_model(model_path, compile=False)

    word2idx = {str(token): int(index) for token, index in cfg["word2idx"].items()}
    maxlen = int(cfg["maxlen"])
    if model.input_shape[-1] != maxlen:
        raise ArtifactError(
            f"Metadata/model mismatch: maxlen={maxlen}, model expects {model.input_shape[-1]}"
        )
    if model.output_shape[-1] != len(classes):
        raise ArtifactError(
            f"Metadata/model mismatch: {len(classes)} classes, model outputs {model.output_shape[-1]}"
        )

    return {
        "model": model,
        "word2idx": word2idx,
        "maxlen": maxlen,
        "classes": list(classes),
        "stopwords": set(stopwords),
    }


def predict_proba(texts: Iterable[object], artifacts: dict) -> np.ndarray:
    """Return one probability vector per input article."""

    text_list = list(texts)
    sequences = np.asarray(
        [
            encode(
                preprocess(text, artifacts["stopwords"]),
                artifacts["word2idx"],
                artifacts["maxlen"],
            )
            for text in text_list
        ],
        dtype="int32",
    )
    if not len(text_list):
        return np.empty((0, len(artifacts["classes"])), dtype="float32")
    return artifacts["model"].predict(sequences, verbose=0)
