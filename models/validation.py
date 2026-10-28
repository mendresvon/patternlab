"""Check training examples and model responses."""
import json
import re
from pathlib import Path
from .data import UUID4
from .model1 import CATEGORIES

TRIPLET_VERSION = 2


KEYS = ("triplet_version", "record_id", "source", "text", "category", "span", "corrected_text",
        "teacher_corrected_text", "generation")


GENERATION_KEYS = ("explanation_en", "explanation_zh", "edit")


SPLITS = ("train", "validation")


CJK = re.compile(r"[\u3400-\u9fff]")


LATIN = re.compile(r"[A-Za-z]")


EN_MAX_WORDS = 30


ZH_MAX_CJK = 60


SIMPLIFIED = set("这们说个为时会对过还没学语词动么吗应该发现实东习题种样数单汉国书读写变")


MAINLAND_TERMS = ("主語", "謂語", "不定式")


SPACED_PUNCTUATION = re.compile(r"\s[.,!?;:]")


def _offsets(span):
    return (isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span))


def expected_replacement(text, span, corrected):
    "Find one replacement at the given span that produces the corrected text."
    start, end = span
    tail = len(text) - end
    if len(corrected) < start + tail:
        return None
    if corrected[:start] != text[:start] or corrected[len(corrected) - tail:] != text[end:]:
        return None
    return corrected[start:len(corrected) - tail]


def spacing_problem(text, span, replacement):
    "Reject spacing-only edits, joined words, and new invalid spacing."
    start, end = span
    fixed = text[:start] + replacement + text[end:]
    if any(c.isspace() and c != " " for c in replacement):
        return True
    if "".join(fixed.split()) == "".join(text.split()):
        return True
    if (fixed[:1].isspace() and not text[:1].isspace()) or (fixed[-1:].isspace() and not text[-1:].isspace()):
        return True
    left = start > 0 and text[start - 1].isalnum()
    right = end < len(text) and text[end].isalnum()
    if replacement:
        # A join is new only where the replaced text did not already touch its neighbour ("do" in "don't" did).
        glued = ((left and replacement[0].isalnum() and not (start < end and text[start].isalnum()))
                 or (right and replacement[-1].isalnum() and not (start < end and text[end - 1].isalnum())))
    else:
        glued = left and right and not text[start].isalnum() and not text[end - 1].isalnum()
    return (glued or fixed.count("  ") > text.count("  ")
            or len(SPACED_PUNCTUATION.findall(fixed)) > len(SPACED_PUNCTUATION.findall(text)))


def needs_adjacent_space(text, span, corrected):
    "Check whether the edit needs a space outside the marked span."
    start, end = span
    wider = ([[start, end + 1]] if text[end:end + 1] == " " else []) + ([[start - 1, end]] if start and
                                                                       text[start - 1] == " " else [])
    return any(expected_replacement(text, w, corrected) is not None for w in wider)


def explanation_problems(en, zh):
    "Check explanation length, language, Taiwan terms, and valid UTF-8 text."
    problems = []
    if not isinstance(en, str) or not en.strip():
        problems.append("explanation_en_blank")
    else:
        if len(en.split()) > EN_MAX_WORDS:
            problems.append("explanation_en_too_long")
        if 2 * len(CJK.findall(en)) > len(LATIN.findall(en)) or not LATIN.search(en):
            problems.append("explanation_en_not_english")
    if not isinstance(zh, str) or not zh.strip():
        problems.append("explanation_zh_blank")
    elif not CJK.search(zh):
        problems.append("explanation_zh_not_chinese")
    else:
        if SIMPLIFIED & set(zh):
            problems.append("explanation_zh_simplified")
        if any(term in zh for term in MAINLAND_TERMS):
            problems.append("explanation_zh_mainland_term")
        if len(CJK.findall(zh)) > ZH_MAX_CJK:
            problems.append("explanation_zh_too_long")
    texts = [s for s in (en, zh) if isinstance(s, str)]
    if any("\n" in s or "\r" in s for s in texts):
        problems.append("explanation_multiline")
    try:
        "".join(texts).encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate, which a JSON escape such as \ud800 can carry
        problems.append("explanation_not_utf8")
    return problems


def generation_problems(generation):
    "Check field order, explanations, and one edit with valid integer positions."
    if not isinstance(generation, dict):
        return ["generation_missing"]
    if tuple(generation) != GENERATION_KEYS:
        return ["generation_keys" if set(generation) != set(GENERATION_KEYS) else "generation_key_order"]
    en, zh, edit = (generation[k] for k in GENERATION_KEYS)
    problems = explanation_problems(en, zh)
    if (not isinstance(edit, dict) or set(edit) != {"span", "replacement"} or not _offsets(edit["span"])
            or min(edit["span"]) < 0 or not isinstance(edit["replacement"], str)):
        problems.append("edit_malformed")
    return problems


def check(triplet, split, drafted=True):
    "Validate one training or validation triplet. Return an empty list on success."
    if not isinstance(triplet, dict) or set(triplet) != set(KEYS):
        return ["bad_keys"]
    if triplet["triplet_version"] != TRIPLET_VERSION or type(triplet["triplet_version"]) is not int:
        return ["bad_version"]
    if not isinstance(triplet["record_id"], str) or not UUID4.fullmatch(triplet["record_id"]):
        return ["bad_record_id"]
    source = triplet["source"]
    if not isinstance(source, dict) or set(source) != {"dataset", "split"}:
        return ["bad_source"]
    text, span, corrected = triplet["text"], triplet["span"], triplet["corrected_text"]
    if not isinstance(text, str) or not text.strip() or not isinstance(corrected, str):
        return ["bad_text"]
    teacher = triplet["teacher_corrected_text"]
    if teacher is not None and (not isinstance(teacher, str) or teacher == corrected):
        return ["bad_teacher_corrected_text"]  # null when the lesson form is the teacher's own

    problems = []
    if source["split"] not in SPLITS:
        problems.append("held_out_split")
    elif source["split"] != split:
        problems.append("wrong_split")
    if source["dataset"] not in ("efcamdat", "efcamdat_synthetic"):
        problems.append("bad_dataset")
    elif split == "validation" and source["dataset"] != "efcamdat":
        problems.append("synthetic_in_evaluation")
    if triplet["category"] not in CATEGORIES:
        problems.append("unknown_category")
    if not _offsets(span):
        return problems + ["span_malformed"]
    if span[0] > span[1]:
        return problems + ["span_reversed"]
    if span[0] < 0 or span[1] > len(text):
        return problems + ["span_out_of_bounds"]

    # The target itself: one edit at Model 1's span, which changes the sentence and leaves a sentence.
    target = expected_replacement(text, span, corrected)
    if not corrected.strip():
        problems.append("target_blank")
    elif corrected == text:
        problems.append("target_no_op")
    elif target is None:
        problems.append("target_needs_adjacent_space" if needs_adjacent_space(text, span, corrected)
                        else "target_not_one_edit_at_span")
    elif spacing_problem(text, span, target):
        problems.append("target_spacing")

    generation = triplet["generation"]
    if not drafted:
        return problems + ([] if generation is None else ["generation_present"])
    found = generation_problems(generation)
    problems += found
    if found:
        return problems
    edit = generation["edit"]
    if edit["span"] != span:
        problems.append("edit_span_mismatch")
        return problems
    applied = text[:span[0]] + edit["replacement"] + text[span[1]:]
    if applied == text:
        problems.append("edit_no_op")
    elif applied != corrected:
        problems.append("edit_wrong_correction")
    return problems


def unique_pairs(items):
    "Reject repeated JSON keys."
    keys = [k for k, _ in items]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(items)


def refuse_constant(name):
    "Reject non-finite JSON numbers."
    raise ValueError(f"non-finite number {name}")


def numbered_jsonl(path):
    "Read strict JSON records. Errors identify the line without exposing its text."
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield number, json.loads(line, object_pairs_hook=unique_pairs, parse_constant=refuse_constant)
                except json.JSONDecodeError:
                    raise ValueError(f"{path}: invalid JSON on line {number}") from None
                except ValueError as error:
                    raise ValueError(f"{path}: line {number}: {error}") from None


def read_jsonl(path):
    return [row for _, row in numbered_jsonl(path)]

STRUCTURE = {"generation_missing", "generation_keys", "generation_key_order", "edit_malformed"}


SCHEMA_CONTENT = {"explanation_en_blank", "explanation_zh_blank", "explanation_zh_not_chinese"}


FENCE = re.compile(r"\A```(?:json)?\s*\n?(.*?)\n?```\Z", re.S)


def parse(raw):
    "Parse one JSON answer. Reject extra prose, repeated keys, and invalid fields."
    try:
        generation = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=refuse_constant)
    except ValueError:
        fenced = FENCE.match(raw.strip())
        codes = ["invalid_json"]
        if fenced and parse(fenced.group(1))[0] is not None:
            codes.append("json_in_fence")
        return None, codes
    problems = generation_problems(generation)
    codes = [c for c in problems if c in STRUCTURE]
    if not codes and not all(isinstance(generation[k], str) for k in ("explanation_en", "explanation_zh")):
        codes = ["explanation_not_string"]
    if not codes:
        codes = [c for c in problems if c in SCHEMA_CONTENT]
    return (None, codes) if codes else (generation, [])
