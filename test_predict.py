import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import clean_text, extract_skills, format_report  # noqa: E402


def test_extract_skills_finds_known_skills():
    text = "Experienced with Python, Docker, and FastAPI for backend microservices."
    skills = extract_skills(text)
    assert "python" in skills
    assert "docker" in skills
    assert "fastapi" in skills
    assert "microservices" in skills


def test_extract_skills_ignores_unrelated_words():
    text = "I enjoy hiking and painting on weekends."
    skills = extract_skills(text)
    assert skills == set()


def test_format_report_includes_recommendation_when_skills_missing():
    result = {
        "predicted_class": "Medium Match",
        "confidence": 0.72,
        "class_probabilities": {"Weak Match": 0.1, "Medium Match": 0.72, "Strong Match": 0.18},
        "common_skills": ["python", "docker"],
        "missing_skills": ["kubernetes", "rag"],
    }
    report = format_report(result)
    assert "Medium Match" in report
    assert "72.0%" in report
    assert "kubernetes" in report
    assert "Recommendation" in report


def test_format_report_handles_no_missing_skills():
    result = {
        "predicted_class": "Strong Match",
        "confidence": 0.95,
        "class_probabilities": {},
        "common_skills": ["python"],
        "missing_skills": [],
    }
    report = format_report(result)
    assert "already covers" in report


def test_clean_text_consistent_with_preprocessing():
    raw = "## Python & Docker!!"
    cleaned = clean_text(raw)
    assert cleaned == cleaned.lower()
    assert "#" not in cleaned
