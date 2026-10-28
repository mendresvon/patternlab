"""Save and load the trained Model 1 weights."""
from dataclasses import asdict
import torch
from .data import SCHEMA_VERSION
from .model1 import CATEGORIES, Detector, ModelConfig

INPUT_KEYS = ("input_ids", "attention_mask", "content_indices", "content_mask")


def model_inputs(batch):
    return {key: batch[key] for key in INPUT_KEYS}


def save_checkpoint(path, model, optimizer, epoch, seed, *, synthetic=False):
    torch.save({"schema_version": SCHEMA_VERSION, "config": asdict(model.config),
                "categories": list(CATEGORIES), "head_state": {
                    k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")},
                "optimizer_state": optimizer.state_dict(), "epoch": epoch, "seed": seed,
                "synthetic": synthetic}, path)


def load_checkpoint(path, *, device="cpu", local_files_only=False):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["schema_version"] != SCHEMA_VERSION or checkpoint["categories"] != list(CATEGORIES):
        raise ValueError("Checkpoint schema/category mismatch")
    if checkpoint["synthetic"]:
        raise ValueError("Synthetic checkpoints cannot be used for learner predictions")
    model = Detector.from_pretrained(ModelConfig(**checkpoint["config"]), local_files_only=local_files_only)
    missing, unexpected = model.load_state_dict(checkpoint["head_state"], strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise ValueError("Checkpoint is missing model head weights")
    return model.to(device).eval()
