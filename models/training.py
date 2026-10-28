"""Prepare batches and calculate validation metrics for the short notebook."""
import random
from functools import partial

import torch
from torch.utils.data import DataLoader

from .data import SentenceDataset, collate, read_jsonl
from .model1 import decode, multitask_loss
from .checkpoints import model_inputs


def seed_everything(seed):
    """Set the random seeds before model creation and training."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_loaders(train_file, validation_file, tokenizer, batch_size, max_length, seed):
    """Validate both files and build batches with the original shuffle order."""
    train_records = read_jsonl(train_file, "train")
    validation_records = read_jsonl(validation_file, "validation")
    train_texts = {record["text"].strip().casefold() for record in train_records}
    if any(record["text"].strip().casefold() in train_texts for record in validation_records):
        raise ValueError("Train and validation contain overlapping sentences")
    train_data = SentenceDataset(train_records, tokenizer, max_length)
    validation_data = SentenceDataset(validation_records, tokenizer, max_length)
    collator = partial(collate, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), collate_fn=collator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, collate_fn=collator)
    return train_loader, validation_loader


def evaluate(model, loader, device):
    model.eval()
    total_loss = count = tp = fp = fn = correct_detection = positives = correct_category = correct_location = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**model_inputs(batch))
            losses = multitask_loss(outputs, batch)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Non-finite loss")
        prediction = decode(outputs)
        gold = batch["has_error"].bool()
        predicted = prediction["has_error"]
        size = len(gold)
        count += size
        total_loss += losses["total"].item() * size
        tp += (predicted & gold).sum().item()
        fp += (predicted & ~gold).sum().item()
        fn += (~predicted & gold).sum().item()
        correct_detection += (predicted == gold).sum().item()
        positives += gold.sum().item()
        # Conditional head accuracy: evaluated on gold positives, not gated by detection.
        correct_category += ((prediction["category"] == batch["category"]) & gold).sum().item()
        correct_location += ((prediction["start"] == batch["start"]) &
                             (prediction["end"] == batch["end"]) & gold).sum().item()
    return {"loss": total_loss / count, "detection_accuracy": correct_detection / count,
            "detection_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "category_accuracy_on_gold_errors": correct_category / positives if positives else None,
            "localization_exact_on_gold_errors": correct_location / positives if positives else None,
            "examples": count}
