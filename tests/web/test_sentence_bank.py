"""Bank structure/privacy checks with invented sentences, never the private bank."""

import json

from fastapi.testclient import TestClient
import pytest

from web.app import app
from web.categories import CATEGORIES
from web.sentence_bank import load_bank


def invented_bank():
    # Structure fixtures only, not a scientific dataset or category judgments.
    return [{"text": f"I collect invented example number {number}.", "category": category,
             "learner": "private-source-id", "span": [0, 1], "seed_text": "private-reference-text"}
            for number, category in enumerate([c for c in CATEGORIES for _ in range(10)] + [None] * 15)]


def write_bank(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_all_85_sentences_load_without_source_metadata_or_file_changes(tmp_path):
    path = tmp_path / "bank.jsonl"
    write_bank(path, invented_bank())
    before = path.read_bytes()
    groups = load_bank(path)
    assert sum(map(len, groups.values())) == 85
    assert all(len(groups[c]) == 10 for c in CATEGORIES) and len(groups["clean"]) == 15
    assert "private-source-id" not in json.dumps(groups)
    assert "private-reference-text" not in json.dumps(groups)
    assert path.read_bytes() == before


@pytest.mark.parametrize("case", ["count", "unknown_category", "long", "wordy", "blank", "duplicate", "not_text", "bad_unicode"])
def test_bad_bank_is_unavailable_instead_of_partially_exposed(tmp_path, case):
    path = tmp_path / "bank.jsonl"
    rows = invented_bank()
    if case == "count":
        rows.pop()
    elif case == "unknown_category":
        rows[0]["category"] = "other"
    elif case == "duplicate":
        rows[1]["text"] = rows[0]["text"].upper()
    else:
        rows[0]["text"] = {"long": "a" * 501, "wordy": "word " * 16,
                           "blank": "   ", "not_text": 42, "bad_unicode": "\ud800"}[case]
    write_bank(path, rows)
    assert load_bank(path) == {}


def test_missing_malformed_and_oversized_files_are_safe(tmp_path):
    assert load_bank(None) == {} and load_bank(tmp_path / "missing.jsonl") == {}
    path = tmp_path / "bank.jsonl"
    for content in ("not json", "[1,2]", " " * 131073):
        path.write_text(content, encoding="utf-8")
        assert load_bank(path) == {}


def test_startup_loads_bank_and_html_escapes_only_public_sentences(tmp_path, monkeypatch):
    path = tmp_path / "bank.jsonl"
    rows = invented_bank()
    rows[0]["text"] = '<img src=x onerror=alert(1)> I watch birds.'
    write_bank(path, rows)
    monkeypatch.setattr(app.state, "bank_path", path)
    monkeypatch.setattr(app.state, "service_options", None)
    monkeypatch.setattr(app.state, "mode", "live")
    with TestClient(app) as client:
        html = client.get("/writing").text
        assert "示範句庫 · 85 句" in html
        assert html.count("<option value=") == 86
        assert "&lt;img" in html and "<img src=x" not in html
        assert "private-source-id" not in html and "private-reference-text" not in html
        assert client.get("/health").json()["sentence_bank_count"] == 85
        assert client.get("/data/interim/bank/bank.jsonl").status_code == 404
    assert app.state.sentence_bank == {}
