"""CPU-only service integration with invented sentences and the shared service fakes."""

import json
import logging
from types import SimpleNamespace
import sys
import threading
import time

from fastapi.testclient import TestClient
import pytest

from web.app import app
from web.categories import CATEGORIES
from web.diagnosis import SAMPLES
from web import service_adapter
from web.service_adapter import call_service, configured_service, service_lock

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_app(monkeypatch):
    monkeypatch.setattr(app.state, "mode", "live")
    monkeypatch.setattr(app.state, "reviewer", False)
    monkeypatch.setattr(app.state, "service", None)
    monkeypatch.setattr(app.state, "service_ready", False)
    monkeypatch.setattr(app.state, "service_options", None)
    monkeypatch.setattr(app.state, "background_model_load", False)
    monkeypatch.setattr(app.state, "sentence_bank", {})
    monkeypatch.setattr(app.state, "bank_path", None)


@pytest.fixture
def service_module():
    return pytest.importorskip("models.service", reason="Shared service/research dependencies unavailable")


def install(monkeypatch, service):
    monkeypatch.setattr(app.state, "service", service)
    monkeypatch.setattr(app.state, "service_ready", True)


@pytest.mark.parametrize("category,text,target,replacement,insertion", [
    ("article_omission", "Nora is illustrator.", "illustrator", "an ", True),
    ("plural_marking_omission", "Two lantern hang outside.", "lantern", "lanterns", False),
    ("past_tense_not_inflected", "Yesterday I paint a lantern.", "paint", "painted", False),
    ("subject_verb_agreement", "Mina paint lanterns every week.", "paint", "paints", False),
    ("be_verb_omission", "The lantern bright.", "bright", "is ", True),
    ("to_omission", "We hope explore the garden.", "explore", "to ", True),
    ("degree_adverb_misuse", "🌿 I very enjoy quiet mornings.", "very", "really", False),
])
def test_all_seven_live_categories_map_validated_prediction_and_omit_english(
        monkeypatch, service_module, category, text, target, replacement, insertion):
    start = text.index(target)
    span = [start, start if insertion else start + len(target)]
    predictor = service_module.FakePredictor.positive(category, span)
    generator = service_module.FakeGenerator(replacement=replacement)
    service = service_module.DiagnosisService(predictor, generator)
    install(monkeypatch, service)
    response = client.post("/api/diagnose", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok" and data["mode"] == "live"
    assert data["category"] == category and data["edit"]["span"] == span
    assert data["location_kind"] == ("insertion" if insertion else "span")
    assert data["corrected_text"] == text[:span[0]] + replacement + text[span[1]:]
    assert "explanation_en" not in data
    assert generator.EXPLANATION_EN not in response.text
    assert generator.calls == [(text, category, span)]
    assert client.get("/health").json()["diagnosis_categories"] == sorted(CATEGORIES)
    assert "即時診斷已就緒" in client.get("/writing").text
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("case,status,reason,http,generated", [
    ("negative", "no_target_error", None, 200, False),
    ("uncertain", "abstained", "uncertain_diagnosis", 200, False),
    ("misaligned", "generation_failed", "invalid_prediction", 503, False),
    ("unsupported", "unsupported_prediction", "unsupported_category", 200, False),
    ("too_long", "unsupported_prediction", "input_too_long", 200, False),
    ("unavailable", "generation_failed", "generation_unavailable", 503, True),
    ("invalid_generation", "generation_failed", "invalid_generation", 503, True),
])
def test_service_statuses_remain_distinct_and_failures_have_no_correction(
        monkeypatch, service_module, case, status, reason, http, generated):
    predictor = service_module.FakePredictor.positive("degree_adverb_misuse", [2, 6], aligned=case != "misaligned")
    if case == "negative":
        predictor = service_module.FakePredictor()
    elif case == "unsupported":
        predictor = service_module.FakePredictor.positive("not_supported", [2, 6])
    elif case == "too_long":
        predictor = service_module.FakePredictor(service_module.InputTooLong())
    generator = service_module.FakeGenerator(
        RuntimeError("private generated text") if case == "unavailable" else
        "private malformed generation" if case == "invalid_generation" else None)
    service = service_module.DiagnosisService(predictor, generator, min_probability=0.95 if case == "uncertain" else None)
    install(monkeypatch, service)
    response = client.post("/api/diagnose", json={"text": "I very enjoy quiet mornings."})
    data = response.json()
    assert response.status_code == http
    assert (data["status"], data["reason_code"]) == (status, reason)
    assert all(data[field] is None for field in ("category", "edit", "corrected_text", "explanation_zh", "location_kind"))
    assert "explanation_en" not in data and "private" not in response.text
    assert bool(generator.calls) == generated


def test_direct_input_too_long_exception_is_friendly(monkeypatch, service_module):
    class Raises:
        def diagnose_with_prediction(self, text):
            raise service_module.InputTooLong()

    install(monkeypatch, Raises())
    data = client.post("/api/diagnose", json={"text": "A short invented sentence."}).json()
    assert data["status"] == "unsupported_prediction" and data["reason_code"] == "input_too_long"


@pytest.mark.parametrize("changes", [
    {"extra": "unexpected"}, {"error_probability": float("nan")},
    {"span": [2, 999]}, {"location_kind": "insertion"},
])
def test_invalid_model_predictions_do_not_blame_valid_learner_input(monkeypatch, service_module, changes):
    predictor = service_module.FakePredictor.positive("degree_adverb_misuse", [2, 6])
    predictor.result.update(changes)
    generator = service_module.FakeGenerator()
    install(monkeypatch, service_module.DiagnosisService(predictor, generator))
    text = "I very enjoy quiet mornings."
    response = client.post("/api/diagnose", json={"text": text})
    data = response.json()
    assert response.status_code == 503
    assert (data["status"], data["reason_code"]) == ("generation_failed", "invalid_prediction")
    assert data["original_text"] == text
    assert all(data[field] is None for field in ("category", "edit", "corrected_text", "explanation_zh"))
    assert generator.calls == []


def test_english_requires_launch_setting_in_html_and_json(monkeypatch, service_module):
    service = service_module.DiagnosisService(
        service_module.FakePredictor.positive("degree_adverb_misuse", [2, 6]), service_module.FakeGenerator())
    install(monkeypatch, service)
    text = "I very enjoy quiet mornings."
    assert "diagnosis-explanation-en" not in client.get("/writing?reviewer=true").text
    assert "explanation_en" not in client.post("/api/diagnose?reviewer=true", json={"text": text}).json()
    assert client.post("/api/diagnose", json={"text": text, "reviewer": True}).status_code == 422
    monkeypatch.setattr(app.state, "reviewer", True)
    html = client.get("/writing").text
    assert "審閱模式" in html and 'id="diagnosis-explanation-en"' in html
    assert client.post("/api/diagnose", json={"text": text}).json()["explanation_en"] == service.generator.EXPLANATION_EN


def test_sample_does_not_call_configured_real_service(monkeypatch):
    class MustNotRun:
        def diagnose_with_prediction(self, text):
            pytest.fail("Sample mode called the real service")

    install(monkeypatch, MustNotRun())
    monkeypatch.setattr(app.state, "mode", "sample")
    assert client.post("/api/diagnose", json={"text": SAMPLES["word"]["text"]}).json()["status"] == "ok"
    assert not client.get("/health").json()["model_ready"]


def test_unconfigured_and_bad_service_never_fabricate_results():
    data = call_service("An invented sentence.", None)
    assert data["status"] == "generation_failed" and data["edit"] is None
    assert configured_service(None) == (None, False)
    for output in (None, {}, {"response": {}, "prediction": None}):
        data = call_service("An invented sentence.", SimpleNamespace(diagnose_with_prediction=lambda text: output))
        assert data["status"] == "generation_failed" and data["corrected_text"] is None


def test_loading_seam_calls_shared_loader_once_and_hides_configuration_failures(monkeypatch, capsys):
    seen = []
    fake = object()  # Loader sentinel only; never used as an inference implementation.

    def loader(**options):
        seen.append(options)
        return fake

    monkeypatch.setitem(sys.modules, "models.service", SimpleNamespace(load_service=loader))
    options = {"checkpoint": "private/checkpoint.pt", "adapter": "private/adapter", "device": "cpu"}
    assert configured_service(options) == (fake, True)
    assert seen == [options]

    def broken(**options):
        raise RuntimeError("private path and credentials")

    monkeypatch.setitem(sys.modules, "models.service", SimpleNamespace(load_service=broken))
    service, ready = configured_service(options)
    assert service is None and not ready
    logged = capsys.readouterr().err
    assert "RuntimeError" in logged and "private" not in logged


def test_concurrent_request_waits_for_the_first_model_call(monkeypatch, service_module):
    predictor = service_module.FakePredictor()
    install(monkeypatch, service_module.DiagnosisService(predictor, service_module.FakeGenerator()))
    service_lock.acquire()
    timer = threading.Timer(0.2, service_lock.release)
    timer.start()
    try:
        response = client.post("/api/diagnose", json={"text": "An invented sentence."})
    finally:
        timer.join()
    assert response.status_code == 200
    assert response.json()["status"] == "no_target_error"
    assert len(predictor.calls) == 1


def test_request_still_busy_after_the_wait_does_not_call_the_model(monkeypatch, service_module):
    predictor = service_module.FakePredictor()
    install(monkeypatch, service_module.DiagnosisService(predictor, service_module.FakeGenerator()))
    monkeypatch.setattr(service_adapter, "BUSY_WAIT_SECONDS", 0.05)
    service_lock.acquire()
    try:
        response = client.post("/api/diagnose", json={"text": "An invented sentence."})
        assert response.status_code == 503
        assert response.json()["reason_code"] == "service_busy"
        assert response.json()["corrected_text"] is None
        assert predictor.calls == []
    finally:
        service_lock.release()


def test_full_wait_queue_answers_busy_at_once(monkeypatch, service_module):
    predictor = service_module.FakePredictor()
    install(monkeypatch, service_module.DiagnosisService(predictor, service_module.FakeGenerator()))
    monkeypatch.setattr(service_adapter, "MAX_WAITING", 0)
    service_lock.acquire()
    try:
        started = time.monotonic()
        response = client.post("/api/diagnose", json={"text": "An invented sentence."})
        assert time.monotonic() - started < service_adapter.BUSY_WAIT_SECONDS / 2
        assert response.json()["reason_code"] == "service_busy"
        assert predictor.calls == []
    finally:
        service_lock.release()
    assert service_adapter.waiting == 0


def test_abandoned_request_skips_the_model_after_waiting(service_module):
    predictor = service_module.FakePredictor()
    service = service_module.DiagnosisService(predictor, service_module.FakeGenerator())
    abandoned = threading.Event()
    abandoned.set()
    data = call_service("An invented sentence.", service, abandoned=abandoned)
    assert data["reason_code"] == "service_busy"
    assert predictor.calls == []
    assert not service_lock.locked() and service_adapter.waiting == 0


def test_model_load_failure_keeps_service_unloaded_and_practice_available(monkeypatch):
    def broken_loader(**options):
        raise RuntimeError("private artifact path")

    monkeypatch.setitem(sys.modules, "models.service",
                        SimpleNamespace(load_service=broken_loader))
    monkeypatch.setattr(app.state, "service_options", {"checkpoint": "one.pt", "adapter": "adapter"})
    with TestClient(app) as running:
        assert app.state.service is None
        assert not running.get("/health").json()["model_ready"]
        assert running.get("/practice").status_code == 200
        response = running.post("/api/diagnose", json={"text": SAMPLES["word"]["text"]})
        assert response.status_code == 503
        assert response.json()["mode"] == "live" and response.json()["corrected_text"] is None
        assert "private artifact path" not in response.text


def test_background_loading_serves_web_but_keeps_diagnosis_unavailable_until_ready(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow_loader(options):
        started.set()
        assert release.wait(5)
        return object(), True

    monkeypatch.setattr(app.state, "background_model_load", True)
    monkeypatch.setattr("web.app.configured_service", slow_loader)
    with TestClient(app) as running:
        try:
            assert started.wait(2)
            assert running.get("/practice").status_code == 200
            assert running.get("/health").json()["model_ready"] is False
            assert running.get("/health").json()["model_loading"] is True
            assert "模型載入中" in running.get("/writing").text
            response = running.post("/api/diagnose", json={"text": "The room is quiet."})
            assert response.status_code == 503 and response.json()["corrected_text"] is None
            assert response.json()["reason_code"] == "model_loading"
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while not running.get("/health").json()["model_ready"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert running.get("/health").json()["model_ready"] is True
        assert running.get("/health").json()["model_loading"] is False
    assert app.state.service is None and app.state.service_ready is False


def test_service_log_keeps_learner_text_and_english_out(monkeypatch, service_module, caplog):
    service = service_module.DiagnosisService(service_module.FakePredictor(), service_module.FakeGenerator())
    install(monkeypatch, service)
    with caplog.at_level(logging.INFO, logger="models.service"):
        client.post("/api/diagnose", json={"text": "A private invented learner sentence."})
    assert "request_id=" in caplog.text
    assert "A private invented learner sentence" not in caplog.text
    assert service.generator.EXPLANATION_EN not in caplog.text


@pytest.mark.parametrize("body", ['{"text":"First","text":"Second"}', '{"text":NaN}',
                                 '{"text":"private"}' + ' ' * 32768])
def test_ambiguous_json_is_rejected_privately(body):
    response = client.post("/api/diagnose", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["original_text"] is None
    assert "private" not in response.text and "First" not in response.text
