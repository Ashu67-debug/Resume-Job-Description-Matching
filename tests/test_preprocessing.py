import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import (  # noqa: E402
    LABEL_MEDIUM,
    LABEL_STRONG,
    LABEL_WEAK,
    build_dataset,
    build_medium_resume,
    clean_text,
)


def test_clean_text_lowercases_and_strips_markdown():
    raw = "## Skills\n- Python\n- **Docker**\n"
    cleaned = clean_text(raw)
    assert cleaned == cleaned.lower()
    assert "#" not in cleaned
    assert "python" in cleaned
    assert "docker" in cleaned


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_clean_text_collapses_whitespace():
    raw = "a    b\n\nc"
    assert clean_text(raw) == "a b c"


def test_build_medium_resume_removes_listed_skills():
    resume = "Skilled in SQL, ERP integration, and BOM systems."
    filtered_info = {"Skills": ["ERP integration"], "Experience": ""}
    result = build_medium_resume(resume, filtered_info)
    assert "erp integration" not in result.lower() or "ERP integration" not in result


def test_build_medium_resume_applies_swap_instruction():
    resume = 'Oracle SCM Developer (2014 - 2017)'
    filtered_info = {
        "Skills": [],
        "Experience": 'instead of "Oracle SCM Developer (2014 - 2017)" use "Oracle SCM Developer (2014 - 2016)"',
    }
    result = build_medium_resume(resume, filtered_info)
    assert "2014 - 2016" in result
    assert "2014 - 2017" not in result


def test_build_medium_resume_applies_drop_instruction():
    resume = "Experience includes Fishing and hunting workers (2017 - 2021) among others."
    filtered_info = {
        "Skills": [],
        "Experience": 'without including "Fishing and hunting workers (2017 - 2021)"',
    }
    result = build_medium_resume(resume, filtered_info)
    assert "Fishing and hunting workers (2017 - 2021)" not in result


def test_build_dataset_produces_three_labels_per_record(tmp_path):
    jsonl_path = tmp_path / "sample.jsonl"
    record = {
        "Job-Description": "We need a Python developer with FastAPI and Docker experience for our backend team.",
        "Resume-matched": "Experienced Python developer skilled in FastAPI, Docker, and cloud deployment for backend systems.",
        "Resume-unmatched": "Marketing specialist with experience in social media campaigns and brand strategy planning.",
        "Filtered-information": {
            "Skills": ["Docker"],
            "Experience": "",
        },
    }
    import json

    with open(jsonl_path, "w") as f:
        f.write(json.dumps(record) + "\n")

    df = build_dataset(jsonl_path, min_words=3)
    assert set(df["match_label"].unique()) <= {LABEL_WEAK, LABEL_MEDIUM, LABEL_STRONG}
    assert len(df) <= 3
    assert list(df.columns) == ["resume_text", "job_description", "match_label"]


def test_build_dataset_filters_short_texts(tmp_path):
    jsonl_path = tmp_path / "sample.jsonl"
    record = {
        "Job-Description": "short jd",
        "Resume-matched": "short resume",
        "Resume-unmatched": "short resume too",
        "Filtered-information": {"Skills": [], "Experience": ""},
    }
    import json

    with open(jsonl_path, "w") as f:
        f.write(json.dumps(record) + "\n")

    df = build_dataset(jsonl_path, min_words=15)
    assert len(df) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
