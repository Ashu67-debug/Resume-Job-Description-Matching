"""
predict.py
==========
Run the trained model on a single (resume, job description) pair and
print a human-readable match report, including simple skill-overlap
diagnostics (common / missing skills) as described in the design doc
(section 16.4).

Usage:
    python src/predict.py \
        --resume "Python ML engineer with FastAPI and Docker experience" \
        --jd "Need AI Engineer with Python FastAPI Docker and RAG pipelines"
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importing model registers the custom AbsDifference / ElementwiseProduct
# layers with Keras so tf.keras.models.load_model can deserialize them.
import model  # noqa: F401

LABEL_NAMES = ["Weak Match", "Medium Match", "Strong Match"]

# A small, illustrative technical-skill vocabulary used only for the
# common/missing skills diagnostic shown alongside the prediction.
# In production this would be replaced by a proper skills taxonomy / NER model.
SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "sql", "nosql", "docker",
    "kubernetes", "aws", "azure", "gcp", "fastapi", "flask", "django",
    "react", "node", "tensorflow", "pytorch", "keras", "rag", "llm",
    "vector database", "vector search", "nlp", "machine learning",
    "deep learning", "spark", "airflow", "terraform", "ci/cd", "microservices",
    "rest api", "graphql", "mongodb", "postgresql", "mysql", "redis",
    "kafka", "linux", "git", "agile", "scrum",
]


def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"#+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9%.,/+#\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> set:
    text = clean_text(text)
    return {skill for skill in SKILL_VOCAB if skill in text}


def load_artifacts(model_path, tokenizer_path, config_path):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(config_path) as f:
        config = json.load(f)
    return model, tokenizer, config


def predict_match(model, tokenizer, max_len, resume_text: str, jd_text: str):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    Xr = pad_sequences(
        tokenizer.texts_to_sequences([resume_clean]), maxlen=max_len,
        padding="post", truncating="post",
    )
    Xj = pad_sequences(
        tokenizer.texts_to_sequences([jd_clean]), maxlen=max_len,
        padding="post", truncating="post",
    )

    probs = model.predict({"resume_tokens": Xr, "jd_tokens": Xj}, verbose=0)[0]
    pred_class = int(np.argmax(probs))

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    common = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)

    return {
        "predicted_class": LABEL_NAMES[pred_class],
        "confidence": float(probs[pred_class]),
        "class_probabilities": {
            LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)
        },
        "common_skills": common,
        "missing_skills": missing,
    }


def format_report(result: dict) -> str:
    lines = [
        f"Predicted class: {result['predicted_class']}",
        f"Confidence: {result['confidence'] * 100:.1f}%",
        f"Common skills: {', '.join(result['common_skills']) or '(none detected)'}",
        f"Missing skills: {', '.join(result['missing_skills']) or '(none detected)'}",
    ]
    if result["missing_skills"]:
        lines.append(
            f"Recommendation: Strengthen resume bullets around "
            f"{', '.join(result['missing_skills'][:3])}."
        )
    else:
        lines.append("Recommendation: Resume already covers the key JD skills.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Predict resume/JD match class for a single pair.")
    parser.add_argument("--resume", type=str, required=True, help="Resume text")
    parser.add_argument("--jd", type=str, required=True, help="Job description text")
    parser.add_argument("--model", type=str, default="models/resume_jd_match_model.keras")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.pkl")
    parser.add_argument("--model-config", type=str, default="models/model_config.json")
    parser.add_argument("--output-md", type=str, default=None, help="Optional path to write a Markdown report")
    args = parser.parse_args()

    model, tokenizer, config = load_artifacts(args.model, args.tokenizer, args.model_config)
    result = predict_match(model, tokenizer, config["max_len"], args.resume, args.jd)
    report = format_report(result)
    print(report)

    if args.output_md:
        out_path = Path(args.output_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("# Resume-JD Match Prediction\n\n```\n" + report + "\n```\n")
        print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
