"""Validate Model 1, call Model 2, and return one checked result."""
import hashlib
import json
import logging
import math
from pathlib import Path
import uuid
from .model1 import CATEGORIES
from .model2 import MODEL, MODEL_REVISION, PROMPT, PROMPT_VERSION, answer, adapter_problems, revision_problem, trained_adapter
from .validation import _offsets, spacing_problem, parse

CONTRACT_VERSION = 1


PREDICTION_KEYS = ("has_error", "error_probability", "category", "span", "location_kind")


REQUEST_ID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")


log = logging.getLogger(__name__)


class InputTooLong(Exception):
    "The sentence exceeds the token limit. The service does not cut it."


def response(request_id, status, reason_code=None, generation=None, corrected_text=None):
    "Build a response. Include an edit and explanations only on success."
    return {"contract_version": CONTRACT_VERSION, "request_id": request_id, "status": status,
            "explanation_en": generation["explanation_en"] if generation else None,
            "explanation_zh": generation["explanation_zh"] if generation else None,
            "corrected_text": corrected_text,
            "edit": {"span": list(generation["edit"]["span"]), "replacement": generation["edit"]["replacement"]}
            if generation else None,
            "reason_code": reason_code}


def valid_request_id(request_id):
    return (isinstance(request_id, str) and 1 <= len(request_id) <= 64 and set(request_id) <= REQUEST_ID_CHARS
            and request_id[0] not in "._-")


def offsets(span):
    return _offsets(span) and min(span) >= 0


def prediction_failure(text, prediction):
    "Return the first failed prediction check. Token alignment is checked separately."
    if isinstance(prediction, (list, tuple)):
        # Never coerce an array of diagnoses into one. an array of anything else is malformed.
        diagnoses = len(prediction) > 1 and all(isinstance(p, dict) and type(p.get("has_error")) is bool
                                                for p in prediction)
        return ("unsupported_prediction", "multiple_diagnoses") if diagnoses else ("invalid_input", "schema_violation")
    if not isinstance(prediction, dict):
        return "invalid_input", "schema_violation"
    probability = prediction.get("error_probability")
    if isinstance(probability, float) and not math.isfinite(probability):
        return "invalid_input", "invalid_json"
    # The adapter drops predict.predict's static `scope` display string (contract, Required input).
    if set(prediction) - {"scope"} != set(PREDICTION_KEYS):
        return "invalid_input", "schema_violation"
    has_error, category, span, kind = (prediction[k] for k in ("has_error", "category", "span", "location_kind"))
    if (type(has_error) is not bool or isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1 or not (category is None or isinstance(category, str))
            or not (span is None or offsets(span)) or kind not in ("span", "insertion", None)):
        return "invalid_input", "schema_violation"
    if not has_error:
        return None if (category, span, kind) == (None, None, None) else ("invalid_input", "schema_violation")
    if category is None or span is None or kind is None:
        return "invalid_input", "schema_violation"
    if category not in CATEGORIES:
        return "unsupported_prediction", "unsupported_category"
    start, end = span
    if not start <= end <= len(text):
        return "invalid_input", "invalid_span"
    if kind != ("insertion" if start == end else "span"):
        return "invalid_input", "inconsistent_prediction"
    return None


def checked_generation(text, span, raw):
    "Validate the JSON and one edit at the predicted span. Return the answer and corrected text."
    if not isinstance(raw, str):
        return None
    generation, _ = parse(raw)
    if generation is None or generation["edit"]["span"] != list(span):
        return None
    start, end = span
    replacement = generation["edit"]["replacement"]
    corrected = text[:start] + replacement + text[end:]
    if corrected == text or not any(c.isalnum() for c in corrected) or spacing_problem(text, span, replacement):
        return None
    # The contract rejects excessive rewriting: a one-word insertion, or the span's words plus at most one.
    if len(replacement.split()) > len(text[start:end].split()) + 1:
        return None
    try:
        "".join((generation["explanation_en"], generation["explanation_zh"], replacement)).encode("utf-8")
    except UnicodeEncodeError:
        return None
    return generation, corrected


class DiagnosisService:
    "Run both models. An optional minimum probability can make the service abstain."

    def __init__(self, predictor, generator, *, min_probability=None):
        self.predictor = predictor
        self.generator = generator
        self.min_probability = min_probability

    def diagnose(self, text: str, request_id=None) -> dict:
        "Return one response. Create a safe request ID when none is supplied."
        return self.diagnose_with_prediction(text, request_id)["response"]

    def diagnose_with_prediction(self, text: str, request_id=None) -> dict:
        "Return the response and validated prediction separately for website highlighting."
        result, prediction = self._route(text, request_id)
        log.info("diagnosis request_id=%s status=%s reason_code=%s", result["request_id"], result["status"],
                 result["reason_code"])
        return {"response": result, "prediction": prediction}

    def _route(self, text, request_id):
        "Check the input, route valid positive predictions, and validate the generated answer."
        if request_id is None:
            request_id = uuid.uuid4().hex
        elif not valid_request_id(request_id):
            return response(None, "invalid_input", "schema_violation"), None
        if not isinstance(text, str) or not text.strip():
            return response(request_id, "invalid_input", "schema_violation"), None
        try:
            prediction = self.predictor.predict(text)
        except InputTooLong:
            return response(request_id, "unsupported_prediction", "input_too_long"), None
        if failure := prediction_failure(text, prediction):
            return response(request_id, *failure), None
        prediction = {k: prediction[k] for k in PREDICTION_KEYS}
        if not prediction["has_error"]:
            return response(request_id, "no_target_error"), prediction
        category, span = prediction["category"], list(prediction["span"])
        if not self.predictor.aligned(text, span):
            return response(request_id, "invalid_input", "invalid_alignment"), None
        if self.min_probability is not None and prediction["error_probability"] < self.min_probability:
            return response(request_id, "abstained", "uncertain_diagnosis"), prediction
        try:
            raw = self.generator.generate(text, category, span)
        except Exception:  # A timeout, an out-of-memory error or a missing model: the contract has one code for all.
            log.warning("generation unavailable for request_id=%s", request_id)
            return response(request_id, "generation_failed", "generation_unavailable"), prediction
        checked = checked_generation(text, span, raw)
        if checked is None:
            return response(request_id, "generation_failed", "invalid_generation"), prediction
        return response(request_id, "ok", generation=checked[0], corrected_text=checked[1]), prediction


class Model1Predictor:
    "Predict with Model 1 and use the training tokenizer rules to check alignment."

    def __init__(self, model, tokenizer, threshold=0.5):
        self.model, self.tokenizer, self.threshold = model, tokenizer, threshold

    def predict(self, text):
        from .predict import predict
        # The same count `data.SentenceDataset` refuses above, checked here so the service never truncates.
        if len(self.tokenizer(text, truncation=False)["input_ids"]) > self.model.config.max_length:
            raise InputTooLong
        return predict(self.model, self.tokenizer, text, self.threshold)

    def aligned(self, text, span):
        from .data import SentenceDataset
        # Any category does: alignment depends on the span only.
        record = {"text": text, "has_error": True, "category": CATEGORIES[0], "span": list(span)}
        try:
            SentenceDataset([record], self.tokenizer, self.model.config.max_length)
        except ValueError:
            return False
        return True


class GemmaGenerator:
    "Generate an explanation and correction with Gemma and the trained adapter."

    def __init__(self, gemma, max_new_tokens=512):
        self.gemma, self.max_new_tokens = gemma, max_new_tokens

    def generate(self, text, category, span):
        from .model2 import answer_batch
        return answer_batch(self.gemma, [(text, category, span)], self.max_new_tokens)[0]


def load_service(checkpoint, adapter, *, device="cpu", threshold=0.5, model=MODEL, revision=None, quantize=True,
                 max_new_tokens=512, local_files_only=False, min_probability=None):
    "Check model and prompt compatibility, then load the two models and tokenizer."
    from transformers import AutoTokenizer
    from .model2 import load_gemma
    from .checkpoints import load_checkpoint

    revision = revision or (MODEL_REVISION if model == MODEL else None)
    if problem := revision_problem(model, revision):
        raise ValueError(problem)
    record = {"model": model, "model_revision": revision, "prompt_version": PROMPT_VERSION,
              "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
              "adapter": {"trained": trained_adapter(adapter)}}
    if problems := adapter_problems(record):
        raise ValueError("; ".join(problems))
    detector = load_checkpoint(checkpoint, device=device, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint).parent / "tokenizer", use_fast=True)
    gemma = load_gemma(model, revision, adapter, quantize=quantize, local_files_only=local_files_only)
    return DiagnosisService(Model1Predictor(detector, tokenizer, threshold), GemmaGenerator(gemma, max_new_tokens),
                            min_probability=min_probability)
