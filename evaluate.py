"""
evaluate.py
===========
Loads a trained Resume/JD match model + tokenizer, scores it on the
held-out test split, and writes a confusion-matrix plot plus a metrics
report.

Usage:
    python src/evaluate.py \
        --model models/resume_jd_match_model.keras \
        --tokenizer models/tokenizer.pkl \
        --test-data data/processed/test_split.csv
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importing model registers the custom AbsDifference / ElementwiseProduct
# layers with Keras so tf.keras.models.load_model can deserialize them.
import model  # noqa: F401

LABEL_NAMES = ["Weak", "Medium", "Strong"]


def encode(tokenizer, texts, max_len):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


def plot_confusion_matrix(cm, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Resume-JD Match — Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Resume/JD match model on the test split.")
    parser.add_argument("--model", type=str, default="models/resume_jd_match_model.keras")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.pkl")
    parser.add_argument("--model-config", type=str, default="models/model_config.json")
    parser.add_argument("--test-data", type=str, default="data/processed/test_split.csv")
    parser.add_argument("--confusion-matrix-out", type=str, default="outputs/confusion_matrix.png")
    parser.add_argument("--metrics-out", type=str, default="outputs/evaluation_metrics.json")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)
    with open(args.model_config) as f:
        config = json.load(f)
    max_len = config["max_len"]

    df = pd.read_csv(args.test_data)
    Xr = encode(tokenizer, df["resume_text"], max_len)
    Xj = encode(tokenizer, df["job_description"], max_len)
    y_true = df["match_label"].to_numpy()

    probs = model.predict({"resume_tokens": Xr, "jd_tokens": Xj}, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 (macro):{f1:.4f}")
    print()
    print(report)
    print("Confusion matrix:")
    print(cm)

    plot_confusion_matrix(cm, Path(args.confusion_matrix_out))
    print(f"Saved confusion matrix plot to {args.confusion_matrix_out}")

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_test_examples": int(len(df)),
    }
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_out}")


if __name__ == "__main__":
    main()
