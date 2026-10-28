"""Scripted model responses for tests. These do not run trained models."""
import json
from models.model2 import answer
from models.service import InputTooLong

class FakePredictor:
    """A scripted Model 1 for tests and GPU-free demos. `result` is the prediction to return: a dict, a list (to
    test multiple diagnoses), an exception to raise (such as `InputTooLong()`), or a function of the text. The
    default finds no target error. `aligned` is what the alignment check answers. Calls are kept in `calls`."""

    NEGATIVE = {"has_error": False, "error_probability": 0.1, "category": None, "span": None, "location_kind": None}

    def __init__(self, result=None, *, aligned=True):
        self.result = self.NEGATIVE if result is None else result
        self._aligned = aligned
        self.calls = []

    @classmethod
    def positive(cls, category, span, probability=0.9, **kwargs):
        """A predictor that finds `category` at `span` in every sentence."""
        return cls({"has_error": True, "error_probability": probability, "category": category, "span": list(span),
                    "location_kind": "insertion" if span[0] == span[1] else "span"}, **kwargs)

    def predict(self, text):
        self.calls.append(text)
        if isinstance(self.result, BaseException):
            raise self.result
        result = self.result(text) if callable(self.result) else self.result
        return json.loads(json.dumps(result))  # a copy, so the service cannot change the script

    def aligned(self, text, span):
        return self._aligned


class FakeGenerator:
    """A scripted Gemma. `result` is the raw answer: a string, an exception to raise, or a function of
    (text, category, span). The default writes a valid contract answer at the given span, with `replacement` (or
    a stand-in per category) and fixed invented explanations. Calls are kept in `calls`."""

    REPLACEMENTS = {"article_omission": "the ", "be_verb_omission": "is ", "to_omission": "to ",
                    "degree_adverb_misuse": "really"}
    EXPLANATION_EN = "This is a stand-in explanation from the fake generator."
    EXPLANATION_ZH = "這是假的產生器寫的示範說明。"

    def __init__(self, result=None, *, replacement=None):
        self.result, self.replacement = result, replacement
        self.calls = []

    def generate(self, text, category, span):
        self.calls.append((text, category, list(span)))
        if isinstance(self.result, BaseException):
            raise self.result
        if self.result is not None:
            return self.result(text, category, span) if callable(self.result) else self.result
        start, end = span
        replacement = self.replacement
        if replacement is None:
            # Suffix categories (plural, past tense, agreement) add a stand-in ending to the marked word.
            replacement = self.REPLACEMENTS.get(category, text[start:end] + "s")
        return answer({"explanation_en": self.EXPLANATION_EN, "explanation_zh": self.EXPLANATION_ZH,
                       "edit": {"span": [start, end], "replacement": replacement}})
