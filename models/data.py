"""Private JSONL examples and exact character-to-subword alignment (no truncation)."""
import json
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .model1 import CATEGORIES

SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
UUID4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
RECORD_KEYS = {"record_schema_version", "record_id", "text", "has_error", "category", "span", "source"}


def validate_record(record):
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if type(record.get("has_error")) is not bool:
        raise ValueError("has_error must be a boolean")
    if record["has_error"]:
        if record.get("category") not in CATEGORIES:
            raise ValueError("Unknown category")
        span = record.get("span")
        if (not isinstance(span, list) or len(span) != 2 or
                any(type(x) is not int for x in span) or
                not 0 <= span[0] <= span[1] <= len(text)):
            raise ValueError("span must be [start, end] character offsets")
    elif record.get("category") is not None or record.get("span") is not None:
        raise ValueError("Correct examples must have null category and span")
    return record


def validate_metadata(record, split, challenge=False):
    "Check the version 1 record fields and split labels."
    if set(record) != RECORD_KEYS:
        raise ValueError("Record keys must match dataset schema v1")
    if record["record_schema_version"] != SCHEMA_VERSION or type(record["record_schema_version"]) is not int:
        raise ValueError("Unsupported record_schema_version")
    if not isinstance(record["record_id"], str) or not UUID4.fullmatch(record["record_id"]):
        raise ValueError("record_id must be a canonical lowercase UUIDv4")
    learner = record["source"] in ({"dataset": "efcamdat", "split": split},
                                   {"dataset": "efcamdat_synthetic", "split": split})
    ai = record["source"] == {"dataset": "ai_written", "split": split} and (split == "train" or challenge)
    if not (learner or ai):
        raise ValueError(f"source must be efcamdat or efcamdat_synthetic with split {split!r}, "
                         "or ai_written in train or a challenge file")
    return record


class SentenceDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=256):
        self.items = []
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with offset mappings is required")
        seen = set()
        for number, record in enumerate(records, 1):
            try:
                validate_record(record)
                key = record["text"].strip().casefold()
                if key in seen:
                    raise ValueError("Duplicate sentence; v1 allows one focal annotation per sentence")
                seen.add(key)
                encoded = tokenizer(record["text"], return_offsets_mapping=True,
                                    return_special_tokens_mask=True, truncation=False)
                if len(encoded["input_ids"]) > max_length:
                    raise ValueError("Example exceeds max_length; split or exclude it explicitly")
                indices = [i for i, (offset, special) in enumerate(zip(
                    encoded["offset_mapping"], encoded["special_tokens_mask"]))
                    if not special and offset[1] > offset[0]]
                offsets = [tuple(encoded["offset_mapping"][i]) for i in indices]
                if not offsets:
                    raise ValueError("No content tokens")
                start = end = category = -100
                if record["has_error"]:
                    a, b = record["span"]
                    starts = [x[0] for x in offsets] + [offsets[-1][1]]
                    if a == b:
                        if a not in starts:
                            raise ValueError("Insertion must align to a token start or final token end")
                        start = end = starts.index(a)
                    else:
                        ends = [x[1] for x in offsets]
                        if a not in starts[:-1] or b not in ends:
                            raise ValueError("Span must align exactly to token edges")
                        start, end = starts.index(a), len(ends) - ends[::-1].index(b)
                        if end <= start:
                            raise ValueError("Invalid token span")
                    category = CATEGORIES.index(record["category"])
                self.items.append({"input_ids": encoded["input_ids"],
                    "content_indices": indices, "offsets": offsets,
                    "has_error": int(record["has_error"]), "category": category,
                    "start": start, "end": end})
            except (ValueError, KeyError, TypeError) as error:
                # Never put private learner text in exception messages.
                raise ValueError(f"Example {number}: {error}") from error
        if not self.items:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def read_jsonl(path, split, challenge=False):
    "Read one split file and validate each record before token alignment."
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    records, ids = [], set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    record = validate_metadata(validate_record(json.loads(line)), split, challenge)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON on line {line_number}") from None
                except (ValueError, TypeError, AttributeError) as error:
                    raise ValueError(f"Line {line_number}: {error}") from None
                if record["record_id"] in ids:
                    raise ValueError(f"Line {line_number}: duplicate record_id")
                ids.add(record["record_id"])
                records.append(record)
    return records


def collate(items, pad_token_id):
    batch_size = len(items)
    width = max(len(item["input_ids"]) for item in items)
    tokens = max(len(item["content_indices"]) for item in items)
    batch = {"input_ids": torch.full((batch_size, width), pad_token_id, dtype=torch.long),
             "attention_mask": torch.zeros(batch_size, width, dtype=torch.long),
             "content_indices": torch.zeros(batch_size, tokens, dtype=torch.long),
             "content_mask": torch.zeros(batch_size, tokens, dtype=torch.bool)}
    for row, item in enumerate(items):
        length, count = len(item["input_ids"]), len(item["content_indices"])
        batch["input_ids"][row, :length] = torch.tensor(item["input_ids"])
        batch["attention_mask"][row, :length] = 1
        batch["content_indices"][row, :count] = torch.tensor(item["content_indices"])
        batch["content_mask"][row, :count] = True
    for key in ("has_error", "category", "start", "end"):
        batch[key] = torch.tensor([item[key] for item in items], dtype=torch.long)
    return batch


def character_span(offsets, start, end):
    if start == end:
        point = offsets[start][0] if start < len(offsets) else offsets[-1][1]
        return [point, point]
    return [offsets[start][0], offsets[end - 1][1]]
