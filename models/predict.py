"""Predict one error category and character span."""
import torch
from .data import SentenceDataset, character_span, collate
from .model1 import CATEGORIES, decode
from .checkpoints import model_inputs

def predict(model, tokenizer, text, threshold=0.5):
    dataset = SentenceDataset([{"text": text, "has_error": False}], tokenizer,
                              model.config.max_length)
    batch = collate([dataset[0]], tokenizer.pad_token_id)
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        result = decode(model(**model_inputs(batch)), threshold)
    has_error = bool(result["has_error"][0])
    span = character_span(dataset[0]["offsets"], int(result["start"][0]), int(result["end"][0])) if has_error else None
    return {"has_error": has_error, "error_probability": float(result["probability"][0]),
            "category": CATEGORIES[int(result["category"][0])] if has_error else None,
            "span": span, "location_kind": ("insertion" if span[0] == span[1] else "span") if span else None,
            "scope": "One target error per sentence; multiple-error inputs are unevaluated."}
