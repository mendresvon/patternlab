"""Frozen token encoder -> packed Bi-LSTM -> attention and three task outputs."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CATEGORIES = (
    "article_omission", "plural_marking_omission", "past_tense_not_inflected",
    "subject_verb_agreement", "be_verb_omission", "to_omission", "degree_adverb_misuse",
)

# The roberta-base commit every Model 1 run so far used. Runs that recorded no revision get it too.
ENCODER_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


def encoder_revision(name, revision=None):
    "Use the requested encoder revision or the fixed RoBERTa revision."
    return revision or (ENCODER_REVISION if name == ModelConfig.encoder_name else None)


@dataclass
class ModelConfig:
    encoder_name: str = "FacebookAI/roberta-base"
    encoder_revision: str | None = None
    hidden_size: int = 128
    dropout: float = 0.2
    max_length: int = 256


class Detector(nn.Module):
    def __init__(self, encoder, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.lstm = nn.LSTM(encoder.config.hidden_size, self.config.hidden_size,
                            batch_first=True, bidirectional=True)
        width = 2 * self.config.hidden_size
        self.dropout = nn.Dropout(self.config.dropout)
        self.attention = nn.Linear(width, 1)
        self.detection = nn.Linear(width, 1)
        self.category = nn.Linear(width, len(CATEGORIES))
        # A boundary joins the states on either side, including sentence edges.
        self.start = nn.Linear(2 * width, 1)
        self.end = nn.Linear(2 * width, 1)

    @classmethod
    def from_pretrained(cls, config=None, *, local_files_only=False):
        from transformers import AutoModel
        config = config or ModelConfig()
        return cls(AutoModel.from_pretrained(config.encoder_name,
                   revision=encoder_revision(config.encoder_name, config.encoder_revision), local_files_only=local_files_only), config)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Freeze dropout as well as weights.
        return self

    def forward(self, input_ids, attention_mask, content_indices, content_mask):
        with torch.no_grad():
            encoded = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
        states = encoded.gather(1, content_indices.unsqueeze(-1).expand(
            -1, -1, encoded.shape[-1]))
        lengths = content_mask.sum(1)
        if (lengths == 0).any():
            raise ValueError("Sentences must contain at least one content token")
        packed = pack_padded_sequence(states, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        sequence, _ = self.lstm(packed)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True,
                                         total_length=content_mask.shape[1])
        sequence = self.dropout(sequence) * content_mask.unsqueeze(-1)
        scores = self.attention(sequence).squeeze(-1).masked_fill(~content_mask, -torch.inf)
        attention = scores.softmax(-1)
        pooled = (sequence * attention.unsqueeze(-1)).sum(1)
        edge = sequence.new_zeros(sequence.shape[0], 1, sequence.shape[-1])
        boundaries = torch.cat((torch.cat((edge, sequence), 1),
                                torch.cat((sequence, edge), 1)), -1)
        boundary_mask = torch.arange(boundaries.shape[1], device=lengths.device)[None] <= lengths[:, None]
        return {
            "detection_logits": self.detection(pooled).squeeze(-1),
            "category_logits": self.category(pooled),
            "start_logits": self.start(boundaries).squeeze(-1).masked_fill(~boundary_mask, -torch.inf),
            "end_logits": self.end(boundaries).squeeze(-1).masked_fill(~boundary_mask, -torch.inf),
            "boundary_mask": boundary_mask,
            "attention": attention,
        }


def multitask_loss(outputs, batch):
    "Compute all three losses. Category and location use positive examples only."
    detection = nn.functional.binary_cross_entropy_with_logits(
        outputs["detection_logits"], batch["has_error"].float())
    positive = batch["has_error"].bool()
    # Avoid summing masked -inf logits when an entire batch is correct.
    zero = outputs["category_logits"].sum() * 0
    category = localization = zero
    if positive.any():
        category = nn.functional.cross_entropy(outputs["category_logits"][positive],
                                                batch["category"][positive])
        localization = (nn.functional.cross_entropy(outputs["start_logits"][positive], batch["start"][positive])
                        + nn.functional.cross_entropy(outputs["end_logits"][positive], batch["end"][positive])) / 2
    return {"total": detection + category + localization,
            "detection": detection, "category": category, "localization": localization}


def decode(outputs, threshold=0.5):
    "Choose one valid start/end pair. Equal positions mean an insertion."
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    probability = outputs["detection_logits"].sigmoid()
    joint = outputs["start_logits"].unsqueeze(2) + outputs["end_logits"].unsqueeze(1)
    n = joint.shape[1]
    valid = torch.ones(n, n, dtype=torch.bool, device=joint.device).triu()
    joint = joint.masked_fill(~valid, -torch.inf)
    index = joint.flatten(1).argmax(1)
    return {"has_error": probability >= threshold, "probability": probability,
            "category": outputs["category_logits"].argmax(1),
            "start": index // n, "end": index % n}
