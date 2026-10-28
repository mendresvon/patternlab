"""Checkpoint C UI adapter and sample isolation, without model/corpus dependencies."""

import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.categories import CATEGORIES
from web.diagnosis import MAX_TEXT_LENGTH, SAMPLES, diagnose

client = TestClient(app)
SUCCESS_FIELDS = ("category", "corrected_text", "edit", "location_kind", "explanation_zh")


@pytest.fixture(autouse=True)
def live_mode(monkeypatch):
    monkeypatch.setattr(app.state, "mode", "live")


@pytest.mark.parametrize("sample", SAMPLES.values(), ids=SAMPLES)
def test_live_never_uses_sample_matches(sample):
    response = client.post("/api/diagnose", json={"text": sample["text"]})
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "generation_failed"
    assert data["reason_code"] == "generation_unavailable"
    assert data["mode"] == "live"
    assert all(data[field] is None for field in SUCCESS_FIELDS)
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("sample", SAMPLES.values(), ids=SAMPLES)
def test_sample_states_and_exact_codepoint_edits(sample, monkeypatch):
    monkeypatch.setattr(app.state, "mode", "sample")
    response = client.post("/api/diagnose", json={"text": sample["text"]})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == sample["status"]
    assert data["mode"] == "sample"
    assert data["original_text"] == sample["text"]
    assert "explanation_en" not in data
    if data["status"] == "ok":
        start, end = data["edit"]["span"]
        assert type(start) is type(end) is int
        assert 0 <= start <= end <= len(sample["text"])
        assert (start == end) == (data["location_kind"] == "insertion")
        assert data["corrected_text"] == sample["text"][:start] + data["edit"]["replacement"] + sample["text"][end:]
        assert data["corrected_text"] != sample["text"]
        assert data["category"] in CATEGORIES
        assert data["explanation_zh"]
        assert data["reason_code"] is None
    else:
        assert all(data[field] is None for field in SUCCESS_FIELDS)


def test_sample_matches_exact_original_text_only(monkeypatch):
    monkeypatch.setattr(app.state, "mode", "sample")
    for text in ["I wrote a new sentence.", " " + SAMPLES["word"]["text"],
                 SAMPLES["word"]["text"].lower(), "<script>alert('private')</script>"]:
        data = client.post("/api/diagnose", json={"text": text}).json()
        assert data["status"] == "unsupported_prediction"
        assert data["reason_code"] == "sample_not_found"
        assert data["original_text"] == text
        assert all(data[field] is None for field in SUCCESS_FIELDS)
    assert client.post("/api/diagnose?mode=live", json={"text": SAMPLES["word"]["text"]}).json()["mode"] == "sample"


@pytest.mark.parametrize("payload", [
    {}, {"text": ""}, {"text": "  \n\t"}, {"text": None}, {"text": 42}, {"text": True},
    {"text": "a" * (MAX_TEXT_LENGTH + 1)}, {"text": "😀" * (MAX_TEXT_LENGTH + 1)},
    {"text": "private sentence", "mode": "sample"}, {"text": "private sentence", "extra": True},
    [],
])
def test_invalid_input_is_fixed_and_never_echoes_body(payload):
    response = client.post("/api/diagnose", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "invalid_input"
    assert data["original_text"] is None
    assert all(data[field] is None for field in SUCCESS_FIELDS)
    assert "private sentence" not in response.text
    assert "explanation_en" not in response.text


def test_malformed_json_and_exact_length_boundary():
    # Send the malformed scalar as escaped JSON; httpx correctly refuses to encode
    # a literal lone surrogate before it reaches the server.
    for body in ('{ "text":', '{"text":"\\ud800"}'):
        response = client.post("/api/diagnose", content=body, headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.json()["status"] == "invalid_input"
    original = "😀" * MAX_TEXT_LENGTH
    response = client.post("/api/diagnose", json={"text": original})
    assert response.status_code == 503
    assert response.json()["original_text"] == original
    original = "  A small example.\n"
    assert client.post("/api/diagnose", json={"text": original}).json()["original_text"] == original


def test_launch_mode_banner_is_sitewide_and_not_request_controlled(monkeypatch):
    for path in ("/", "/writing", "/practice", "/practice/article_omission", "/missing"):
        live = client.get(path).text
        assert "非即時模型推論" not in live
    live = client.get("/writing?mode=sample").text
    assert "data-sample-text=" not in live
    assert client.post("/api/diagnose?mode=sample", json={"text": SAMPLES["word"]["text"]}).status_code == 503
    monkeypatch.setattr(app.state, "mode", "sample")
    for path in ("/", "/writing", "/practice", "/practice/article_omission", "/missing"):
        html = client.get(path).text
        assert "示範模式" in html
        assert "顯示預先編寫的範例，非即時模型推論" in html
    html = client.get("/writing").text
    assert html.count("data-sample-text=") == len(SAMPLES)
    assert "explanation_en" not in html
    assert "explanation_zh" not in html  # Results arrive only after a submission.
    assert client.get("/health").json()["mode"] == "sample"
    assert client.get("/health").json()["model_ready"] is False


def test_no_unknown_mode_fallback():
    with pytest.raises(ValueError):
        diagnose(SAMPLES["word"]["text"], "unknown")
