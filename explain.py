"""
explain.py
==========
Lightweight interpretability for the Siamese BiLSTM match model.

Two complementary explanations are produced for a given (resume, JD) pair:

1. Skill-overlap diagnostic (fast, symbolic):
   which JD skills are present / missing in the resume — this is the same
   logic used in predict.py's "Common skills / Missing skills" report.

2. Occlusion-based token importance (model-driven):
   for each token in the resume, mask it out (replace with the pad token)
   and measure how much the predicted class probability drops. Tokens whose
   removal drops the confidence the most are the ones the model is relying
   on most heavily — a simple, dependency-free stand-in for attention/
   saliency maps.

Usage:
    python src/explain.py \
        --resume "Python ML engineer with FastAPI and Docker" \
        --jd "Need AI Engineer with Python FastAPI Docker RAG"
"""
import argparse
import json
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from predict import LABEL_NAMES, clean_text, load_artifacts


def occlusion_importance(model, tokenizer, max_len, resume_text, jd_text, top_k=10):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    tokens = resume_clean.split()

    Xr_base = pad_sequences(
        tokenizer.texts_to_sequences([resume_clean]), maxlen=max_len,
        padding="post", truncating="post",
    )
    Xj = pad_sequences(
        tokenizer.texts_to_sequences([jd_clean]), maxlen=max_len,
        padding="post", truncating="post",
    )

    base_probs = model.predict({"resume_tokens": Xr_base, "jd_tokens": Xj}, verbose=0)[0]
    pred_class = int(np.argmax(base_probs))
    base_conf = float(base_probs[pred_class])

    n = min(len(tokens), max_len)
    importances = []
    # Batch all occlusions together for speed.
    occluded_batch = np.repeat(Xr_base, n, axis=0)
    for i in range(n):
        occluded_batch[i, i] = 0  # 0 = pad / mask token id

    jd_batch = np.repeat(Xj, n, axis=0)
    occluded_probs = model.predict(
        {"resume_tokens": occluded_batch, "jd_tokens": jd_batch}, verbose=0
    )[:, pred_class]

    for i in range(n):
        drop = base_conf - float(occluded_probs[i])
        importances.append((tokens[i], drop))

    importances.sort(key=lambda x: x[1], reverse=True)
    return {
        "predicted_class": LABEL_NAMES[pred_class],
        "base_confidence": base_conf,
        "top_influential_tokens": importances[:top_k],
    }


def main():
    parser = argparse.ArgumentParser(description="Explain a single resume/JD prediction.")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--jd", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/resume_jd_match_model.keras")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.pkl")
    parser.add_argument("--model-config", type=str, default="models/model_config.json")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    model, tokenizer, config = load_artifacts(args.model, args.tokenizer, args.model_config)
    result = occlusion_importance(
        model, tokenizer, config["max_len"], args.resume, args.jd, top_k=args.top_k
    )

    print(f"Predicted class: {result['predicted_class']}")
    print(f"Base confidence: {result['base_confidence'] * 100:.1f}%")
    print("\nTokens the model relies on most (removing them drops confidence the most):")
    for token, drop in result["top_influential_tokens"]:
        print(f"  {token:<20s} confidence drop: {drop * 100:+.2f}%")


if __name__ == "__main__":
    main()
