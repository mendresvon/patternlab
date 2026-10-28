"""Select original practice questions and grade short answers, without any ML imports."""

import json
import random
from pathlib import Path

from .categories import CATEGORIES


def normalize(answer: str) -> str:
    """Ignore outer whitespace and case, but never guess synonyms or spelling."""
    return answer.strip().casefold()


def validate_bank(bank: dict) -> dict:
    """Fail early if the versioned bank cannot supply the promised 4+4 per topic."""
    if bank.get("version") != 1 or not isinstance(bank.get("questions"), list):
        raise ValueError("Expected a version 1 question bank")
    questions = {}
    counts = {category: {"multiple_choice": 0, "fill_blank": 0} for category in CATEGORIES}
    for question in bank["questions"]:
        category, kind = question.get("category"), question.get("type")
        if category not in counts or kind not in counts[category]:
            raise ValueError("Unknown category or question type")
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id or question_id in questions:
            raise ValueError("Question IDs must be nonempty and unique")
        for field in ("prompt_zh", "sentence", "explanation_zh"):
            if not isinstance(question.get(field), str) or not question[field].strip():
                raise ValueError(f"Missing {field}: {question_id}")
        if kind == "multiple_choice":
            options = question.get("options", [])
            if not 2 <= len(options) <= 4:
                raise ValueError(f"Expected 2-4 options: {question_id}")
            ids = [option.get("id") for option in options]
            texts = [option.get("text") for option in options]
            if (len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids)
                    or not all(isinstance(t, str) and t.strip() for t in texts)
                    or len({normalize(t) for t in texts}) != len(texts)
                    or question.get("answer_id") not in ids):
                raise ValueError(f"Invalid options or answer: {question_id}")
        else:
            answers = question.get("accepted_answers")
            if not isinstance(answers, list) or not answers or not all(
                isinstance(answer, str) and normalize(answer) for answer in answers
            ):
                raise ValueError(f"Missing accepted answers: {question_id}")
        counts[category][kind] += 1
        questions[question_id] = question
    if any(count != 4 for types in counts.values() for count in types.values()):
        raise ValueError("Each of seven categories needs four questions of each type")
    return questions


QUESTIONS = validate_bank(json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8")))


def create_round(category: str) -> dict:
    """Return five unique public questions (3 choice + 2 fill), never answer keys."""
    if category not in CATEGORIES:
        raise KeyError(category)
    selected = []
    for kind, count in (("multiple_choice", 3), ("fill_blank", 2)):
        pool = [q for q in QUESTIONS.values() if q["category"] == category and q["type"] == kind]
        selected.extend(random.sample(pool, count))
    random.shuffle(selected)
    public_fields = ("id", "type", "prompt_zh", "sentence", "options")
    return {"category": category, "questions": [
        {key: question[key] for key in public_fields if key in question} for question in selected
    ]}


def check_answer(question_id: str, answer: str) -> dict:
    """Grade against the private bank; reveal feedback only after a submission."""
    question = QUESTIONS[question_id]
    if question["type"] == "multiple_choice":
        options = {option["id"]: option["text"] for option in question["options"]}
        if answer not in options:
            raise ValueError("請選擇題目提供的選項。")
        correct = answer == question["answer_id"]
        accepted = options[question["answer_id"]]
    else:
        correct = normalize(answer) in {normalize(a) for a in question["accepted_answers"]}
        accepted = " / ".join(question["accepted_answers"])
    return {"correct": correct, "correct_answer": accepted, "explanation_zh": question["explanation_zh"]}
