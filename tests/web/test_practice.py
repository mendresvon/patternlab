"""The curated bank, public API boundary, and deterministic grading rules."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.categories import CATEGORIES
from web.practice import QUESTIONS, create_round, validate_bank

client = TestClient(app)
BANK = json.loads(Path(__file__).parents[2].joinpath("web", "questions.json").read_text())


def test_bank_has_56_unique_original_items_and_four_of_each_type():
    assert len(QUESTIONS) == 56
    assert len({q["sentence"] for q in QUESTIONS.values()}) == 56
    for category in CATEGORIES:
        assert Counter(q["type"] for q in QUESTIONS.values() if q["category"] == category) == {
            "multiple_choice": 4, "fill_blank": 4,
        }


@pytest.mark.parametrize("category", CATEGORIES)
def test_round_composition_and_no_early_answers(category):
    seen = set()
    for _ in range(30):
        result = create_round(category)
        questions = result["questions"]
        assert result["category"] == category
        assert len({q["id"] for q in questions}) == 5
        assert Counter(q["type"] for q in questions) == {"multiple_choice": 3, "fill_blank": 2}
        for q in questions:
            assert QUESTIONS[q["id"]]["category"] == category
            assert set(q) <= {"id", "type", "prompt_zh", "sentence", "options"}
            seen.add(q["id"])
    assert seen == {q["id"] for q in QUESTIONS.values() if q["category"] == category}
    response = client.get(f"/api/practice/{category}/round")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert not any(field in response.text for field in ("accepted_answers", "answer_id", "explanation_zh"))


@pytest.mark.parametrize("question", QUESTIONS.values(), ids=QUESTIONS)
def test_every_question_grades_correct_and_incorrect_answers(question):
    if question["type"] == "multiple_choice":
        correct = question["answer_id"]
        incorrect = next(o["id"] for o in question["options"] if o["id"] != correct)
        expected = next(o["text"] for o in question["options"] if o["id"] == correct)
    else:
        correct = f"  {question['accepted_answers'][0].upper()}\t"
        incorrect = "wrong answer"
        expected = " / ".join(question["accepted_answers"])
    for answer, is_correct in ((correct, True), (incorrect, False)):
        response = client.post("/api/practice/check", json={"question_id": question["id"], "answer": answer})
        assert response.status_code == 200
        assert response.json() == {"correct": is_correct, "correct_answer": expected,
                                   "explanation_zh": question["explanation_zh"]}


@pytest.mark.parametrize("payload", [
    {}, {"question_id": "article_01"}, {"question_id": "../questions.json", "answer": "a"},
    {"question_id": "a" * 65, "answer": "a"}, {"question_id": "article_01", "answer": " "},
    {"question_id": "article_01", "answer": "x" * 101}, {"question_id": "article_01", "answer": 1},
    {"question_id": "article_01", "answer": None},
    {"question_id": "article_01", "answer": "a", "correct": True},
    {"question_id": "article_01", "answer": "unknown_option"},
])
def test_malformed_submissions_are_rejected(payload):
    assert client.post("/api/practice/check", json=payload).status_code == 422


def test_unknown_ids_and_bank_privacy():
    assert client.get("/api/practice/unknown/round").status_code == 404
    assert client.post("/api/practice/check", json={"question_id": "unknown", "answer": "a"}).status_code == 404
    for path in ("/questions.json", "/web/questions.json", "/static/questions.json", "/static/../questions.json"):
        assert client.get(path).status_code == 404


def test_no_synonym_or_internal_whitespace_guessing():
    for answer in ("truly", "real ly", "really.", "<script>alert(1)</script>", "😀"):
        result = client.post("/api/practice/check", json={"question_id": "degree_05", "answer": answer})
        assert result.status_code == 200
        assert result.json()["correct"] is False
        assert answer not in result.json()["explanation_zh"]


@pytest.mark.parametrize("change", ["version", "count", "duplicate", "category", "answer", "options", "explanation"])
def test_invalid_bank_fails_at_load(change):
    bank = copy.deepcopy(BANK)
    if change == "version":
        bank["version"] = 2
    elif change == "count":
        bank["questions"].pop()
    elif change == "duplicate":
        bank["questions"][1]["id"] = bank["questions"][0]["id"]
    elif change == "category":
        bank["questions"][0]["category"] = "unknown"
    elif change == "answer":
        bank["questions"][0]["answer_id"] = "unknown"
    elif change == "options":
        bank["questions"][0]["options"][1]["id"] = "a"
    else:
        bank["questions"][0]["explanation_zh"] = ""
    with pytest.raises(ValueError):
        validate_bank(bank)
