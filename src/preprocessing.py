"""
preprocessing.py
=================
Turns the raw Resume/JD JSONL records into a clean, labeled CSV dataset
suitable for training the Siamese BiLSTM match-scorer described in
`Resume_JD_Deep_Neural_Network.ipynb`.

Raw record schema (one JSON object per line):
{
  "Job-Description":   str,   # the job posting
  "Resume-matched":    str,   # a resume written to be a STRONG match for the JD
  "Resume-unmatched":  str,   # a resume written to be a WEAK match for the JD
  "Skills":            list,  # skills relevant to the JD
  "Experiences":       list,
  "Experiences-years": list,
  "Filtered-information": {   # instructions describing a *small* degradation
      "Skills": [...],        # skills to drop from the matched resume
      "Experience": str       # a free-text edit instruction, one of:
                               #   'instead of "<old>" use "<new>"'
                               #   'without including "<text>"'
  }
}

Label design
------------
Each raw record yields **three** training rows:

  label 2 (Strong) -> Job-Description + Resume-matched (as-is)
  label 0 (Weak)   -> Job-Description + Resume-unmatched (as-is)
  label 1 (Medium) -> Job-Description + a *slightly degraded* version of
                       Resume-matched, built by applying the
                       "Filtered-information" edit instructions:
                         - remove the listed skills from the skills section
                         - apply the experience-years / phrase edit

This gives a naturally balanced 3-class dataset without hand-labeling.

Output: data/processed/resume_jd_dataset.csv
Columns: resume_text, job_description, match_label
"""
import argparse
import json
import re
import string
from pathlib import Path

import pandas as pd

LABEL_WEAK, LABEL_MEDIUM, LABEL_STRONG = 0, 1, 2


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, normalize whitespace, strip markdown noise/punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"#+", " ", text)                 # markdown headers
    text = re.sub(r"[-*]{2,}", " ", text)            # markdown bullets/rules
    text = text.lower()
    text = re.sub(r"[^a-z0-9%.,/+#\- ]", " ", text)  # keep tech-relevant chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------
# Medium-match synthesis
# --------------------------------------------------------------------------
def _remove_skills(resume_text: str, skills_to_remove: list) -> str:
    for skill in skills_to_remove:
        if not skill:
            continue
        pattern = re.escape(skill)
        resume_text = re.sub(pattern, "", resume_text, flags=re.IGNORECASE)
    return resume_text


def _apply_experience_edit(resume_text: str, instruction: str) -> str:
    """Apply a free-text edit instruction of one of two known forms."""
    if not instruction:
        return resume_text

    swap = re.match(r'instead of "(.*?)" use "(.*?)"', instruction)
    if swap:
        old, new = swap.group(1), swap.group(2)
        return resume_text.replace(old, new)

    drop = re.match(r'without including "(.*?)"', instruction)
    if drop:
        old = drop.group(1)
        return resume_text.replace(old, "")

    return resume_text


def build_medium_resume(resume_matched: str, filtered_information: dict) -> str:
    """Degrade a strong-match resume slightly to synthesize a medium-match one."""
    text = resume_matched
    fi = filtered_information or {}
    text = _remove_skills(text, fi.get("Skills", []))
    text = _apply_experience_edit(text, fi.get("Experience", ""))
    return text


# --------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------
def load_records(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_dataset(jsonl_path: Path, min_words: int = 15) -> pd.DataFrame:
    rows = []
    for rec in load_records(jsonl_path):
        jd = rec.get("Job-Description", "")
        resume_strong = rec.get("Resume-matched", "")
        resume_weak = rec.get("Resume-unmatched", "")
        resume_medium = build_medium_resume(
            resume_strong, rec.get("Filtered-information", {})
        )

        for resume_text, label in (
            (resume_strong, LABEL_STRONG),
            (resume_medium, LABEL_MEDIUM),
            (resume_weak, LABEL_WEAK),
        ):
            clean_resume = clean_text(resume_text)
            clean_jd = clean_text(jd)
            if len(clean_resume.split()) < min_words or len(clean_jd.split()) < min_words:
                continue
            rows.append(
                {
                    "resume_text": clean_resume,
                    "job_description": clean_jd,
                    "match_label": label,
                }
            )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["resume_text", "job_description"])
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build resume/JD match dataset from raw JSONL.")
    parser.add_argument(
        "--input", type=str, default="data/raw/train_sample.jsonl",
        help="Path to raw JSONL file (Job-Description / Resume-matched / Resume-unmatched schema).",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/resume_jd_dataset.csv",
        help="Path to write the processed CSV dataset.",
    )
    parser.add_argument("--min-words", type=int, default=15)
    args = parser.parse_args()

    df = build_dataset(Path(args.input), min_words=args.min_words)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df):,} rows to {out_path}")
    print(df["match_label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
