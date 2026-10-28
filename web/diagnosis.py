"""Keep sample fixtures isolated from the optional real inference adapter."""

MAX_TEXT_LENGTH = 500  # UI transport limit in Unicode code points, not a tokenizer limit.

# Independently authored examples. Never read corpus/gold/model fixtures for the UI.
SAMPLES = {
    "word": {"label": "修正一個用字", "text": "🌿 I very enjoy quiet mornings.",
             "status": "ok", "category": "degree_adverb_misuse", "target": "very",
             "replacement": "really", "location_kind": "span",
             "explanation_zh": "really 可以直接修飾動詞 enjoy；very 不能直接放在 enjoy 前。中文的「很喜歡」和英文的修飾方式不同，可能讓學習者混淆，但這不代表個人的錯誤一定來自中文。"},
    "insert": {"label": "補上一個字", "text": "🪴 Nora is illustrator.",
               "status": "ok", "category": "article_omission", "target": "illustrator",
               "replacement": "an ", "location_kind": "insertion",
               "explanation_zh": "illustrator 是單數可數名詞，以母音發音開頭，介紹一位插畫家時需要 an。補上 an，句子就成為 Nora is an illustrator。"},
    "negative": {"label": "沒有目標錯誤", "text": "The lantern is very bright.",
                 "status": "no_target_error", "reason_code": None},
    "uncertain": {"label": "結果不確定", "text": "That story felt a little unusual.",
                  "status": "abstained", "reason_code": "uncertain_diagnosis"},
    "scope": {"label": "超出支援範圍", "text": "I draw maps. I collect shells.",
              "status": "unsupported_prediction", "reason_code": "scope_outside_v1"},
    "unavailable": {"label": "修正服務暫停", "text": "We plan visit the aquarium.",
                    "status": "generation_failed", "reason_code": "generation_unavailable"},
}


def result(status: str, text: str | None, mode: str, reason: str | None = None) -> dict:
    """Keep failure/negative responses free of speculative corrections or explanations."""
    return {"status": status, "mode": mode, "original_text": text, "reason_code": reason,
            "category": None, "corrected_text": None, "edit": None,
            "location_kind": None, "explanation_zh": None}


def sample_success(text: str, sample: dict, reviewer=False) -> dict:
    """Construct one exact, code-point-indexed edit for an invented teaching fixture."""
    start = text.index(sample["target"])
    end = start if sample["location_kind"] == "insertion" else start + len(sample["target"])
    payload = result("ok", text, "sample")
    payload.update(category=sample["category"], location_kind=sample["location_kind"],
                   corrected_text=text[:start] + sample["replacement"] + text[end:],
                   edit={"span": [start, end], "replacement": sample["replacement"]},
                   explanation_zh=sample["explanation_zh"])
    if reviewer:
        payload["explanation_en"] = (
            "Really can modify enjoy directly; very cannot. Mandarin expresses degree differently, "
            "but this does not prove the cause of an individual error."
            if sample["location_kind"] == "span" else
            "Illustrator is a singular countable noun starting with a vowel sound, so it needs an. "
            "Adding an makes the sentence Nora is an illustrator.")
    return payload


def diagnose(text: str, mode: str, service=None, reviewer=False, abandoned=None) -> dict:
    """Sample mode matches fixtures only; live calls the shared service through one seam."""
    if mode == "live":
        from .service_adapter import call_service
        return call_service(text, service, reviewer=reviewer, abandoned=abandoned)
    if mode != "sample":
        raise ValueError("Unknown diagnosis mode")
    for sample in SAMPLES.values():
        if text == sample["text"]:
            if sample["status"] == "ok":
                return sample_success(text, sample, reviewer)
            return result(sample["status"], text, mode, sample["reason_code"])
    return result("unsupported_prediction", text, mode, "sample_not_found")
