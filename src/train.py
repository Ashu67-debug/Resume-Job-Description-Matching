"""
train.py
========
End-to-end training script:
  1. loads the processed CSV dataset
  2. fits a shared tokenizer over resumes + JDs
  3. builds the Siamese BiLSTM model (see model.py)
  4. trains with early stopping
  5. saves the trained model, tokenizer, and a training-history plot

Usage:
    python src/train.py \
        --data data/processed/resume_jd_dataset.csv \
        --model-out models/resume_jd_match_model.keras \
        --epochs 15
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from model import ModelConfig, build_siamese_model


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["resume_text", "job_description", "match_label"])
    return df


def fit_tokenizer(texts, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    return tok


def encode(tok: Tokenizer, texts, max_len: int) -> np.ndarray:
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


def plot_history(history, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train accuracy")
    axes[1].plot(history.history["val_accuracy"], label="val accuracy")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle("Resume-JD Siamese BiLSTM — Training History")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the Resume/JD Siamese BiLSTM match model.")
    parser.add_argument("--data", type=str, default="data/processed/resume_jd_dataset.csv")
    parser.add_argument("--model-out", type=str, default="models/resume_jd_match_model.keras")
    parser.add_argument("--tokenizer-out", type=str, default="models/tokenizer.pkl")
    parser.add_argument("--history-plot", type=str, default="outputs/training_history.png")
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    df = load_data(Path(args.data))
    print(f"Loaded {len(df):,} rows")

    # train / val / test split (stratified on label)
    train_df, temp_df = train_test_split(
        df, test_size=args.val_split + args.test_split, stratify=df["match_label"], random_state=args.seed
    )
    rel_test = args.test_split / (args.val_split + args.test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["match_label"], random_state=args.seed
    )
    print(f"Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,}")

    tokenizer = fit_tokenizer(
        pd.concat([train_df["resume_text"], train_df["job_description"]]), args.vocab_size
    )

    def encode_split(d):
        return (
            encode(tokenizer, d["resume_text"], args.max_len),
            encode(tokenizer, d["job_description"], args.max_len),
            d["match_label"].to_numpy(),
        )

    X_res_train, X_jd_train, y_train = encode_split(train_df)
    X_res_val, X_jd_val, y_val = encode_split(val_df)
    X_res_test, X_jd_test, y_test = encode_split(test_df)

    config = ModelConfig(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        lstm_units=args.lstm_units,
    )
    model = build_siamese_model(config)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    history = model.fit(
        {"resume_tokens": X_res_train, "jd_tokens": X_jd_train},
        y_train,
        validation_data=({"resume_tokens": X_res_val, "jd_tokens": X_jd_val}, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Save artifacts
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    print(f"Saved model to {model_out}")

    tok_out = Path(args.tokenizer_out)
    with open(tok_out, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Saved tokenizer to {tok_out}")

    config_out = model_out.parent / "model_config.json"
    with open(config_out, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"Saved model config to {config_out}")

    plot_history(history, Path(args.history_plot))
    print(f"Saved training history plot to {args.history_plot}")

    # Persist held-out test split for evaluate.py
    test_out = Path(args.data).parent / "test_split.csv"
    test_df.to_csv(test_out, index=False)
    print(f"Saved held-out test split to {test_out}")


if __name__ == "__main__":
    main()
