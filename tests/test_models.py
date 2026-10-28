"""Test model behavior without downloading weights or reading learner records."""

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import re
import uuid

import pytest
import torch
from torch import nn

from models import checkpoints, data, model1, model2, service, validation
from models.predict import predict
from tests.fakes import FakeGenerator, FakePredictor

ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(nn.Module):
    """A small fixed embedding for software checks. This is not RoBERTa."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embedding = nn.Embedding(128, 16)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


class TinyTokenizer:
    """Split on spaces so software tests have stable character positions."""

    is_fast = True
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        offsets = [match.span() for match in re.finditer(r"\S+", text)]
        return {"input_ids": [1] + [3 + sum(map(ord, text[a:b])) % 125 for a, b in offsets] + [2],
                "offset_mapping": [(0, 0)] + offsets + [(0, 0)],
                "special_tokens_mask": [1] + [0] * len(offsets) + [1]}


def records():
    return [
        {"text": "Yesterday I polish the lantern.", "has_error": True,
         "category": "past_tense_not_inflected", "span": [12, 18]},
        {"text": "Lina is illustrator.", "has_error": True,
         "category": "article_omission", "span": [8, 8]},
        {"text": "The lantern shines.", "has_error": False, "category": None, "span": None},
    ]


def detector():
    torch.manual_seed(17)
    return model1.Detector(TinyEncoder(), model1.ModelConfig(hidden_size=8, dropout=0))


def batch():
    return data.collate(data.SentenceDataset(records(), TinyTokenizer()).items, pad_token_id=0)


def notebook_definitions(name):
    notebook = json.loads((ROOT / name).read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    return ast.parse(code)


def without_docs(node):
    """Compare executable definitions while allowing clearer explanations."""
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.body:
            first = item.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                item.body = item.body[1:]
    return ast.dump(node, include_attributes=False)


@pytest.mark.parametrize("name,module,names", [
    ("details/model1_training_details.ipynb", model1, ["ModelConfig", "Detector", "multitask_loss", "decode"]),
    ("details/model1_training_details.ipynb", data, ["SentenceDataset", "collate"]),
    ("details/model1_training_details.ipynb", checkpoints, ["save_checkpoint"]),
    ("02_model2_finetuning.ipynb", model2, ["prompt", "answer", "example", "training_examples"]),
])
def test_notebook_definitions_match_shared_runtime(name, module, names):
    tree = notebook_definitions(name)
    for name in names:
        node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        shared = ast.parse(inspect.getsource(getattr(module, name))).body[0]
        assert without_docs(node) == without_docs(shared)


def test_all_notebooks_have_valid_python_and_no_outputs():
    for path in ROOT.rglob("*.ipynb"):
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        ids = [cell["id"] for cell in notebook["cells"]]
        assert len(ids) == len(set(ids))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert not cell["outputs"] and cell["execution_count"] is None
                compile("".join(cell["source"]), str(path), "exec")


def test_model1_training_updates_heads_but_freezes_encoder():
    model = detector()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    before = {name: weight.detach().clone() for name, weight in model.named_parameters()}
    tree = notebook_definitions("details/model1_training_details.ipynb")
    epoch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_epoch")
    scope = {"torch": torch, "model_inputs": checkpoints.model_inputs,
             "multitask_loss": model1.multitask_loss, "decode": model1.decode}
    exec(compile(ast.Module(body=[epoch], type_ignores=[]), "notebook epoch", "exec"), scope)
    metrics = scope["run_epoch"](model, [batch()], "cpu", optimizer)
    assert metrics["examples"] == 3 and metrics["loss"] > 0
    assert not model.encoder.training
    for name, weight in model.named_parameters():
        if name.startswith("encoder."):
            assert torch.equal(weight, before[name]) and weight.grad is None
    assert any(not torch.equal(weight, before[name]) for name, weight in model.named_parameters()
               if not name.startswith("encoder."))


def test_padding_does_not_change_predictions():
    model = detector().eval()
    examples = data.SentenceDataset(records(), TinyTokenizer()).items
    alone = data.collate([examples[2]], 0)
    together = data.collate(examples, 0)
    with torch.no_grad():
        one = model(**checkpoints.model_inputs(alone))
        many = model(**checkpoints.model_inputs(together))
    torch.testing.assert_close(one["detection_logits"][0], many["detection_logits"][2])
    torch.testing.assert_close(one["category_logits"][0], many["category_logits"][2])
    assert many["attention"][2, len(examples[2]["content_indices"]):].sum() == 0


def test_negative_batch_has_only_detection_loss():
    model = detector()
    item = data.SentenceDataset([records()[2]], TinyTokenizer())[0]
    values = data.collate([item], 0)
    losses = model1.multitask_loss(model(**checkpoints.model_inputs(values)), values)
    assert losses["category"].item() == losses["localization"].item() == 0
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_character_alignment_preserves_insertions_and_unicode():
    row = {"text": "🌿 Lina is illustrator.", "has_error": True,
           "category": "article_omission", "span": [10, 10]}
    item = data.SentenceDataset([row], TinyTokenizer())[0]
    assert data.character_span(item["offsets"], item["start"], item["end"]) == [10, 10]
    row["span"] = [11, 11]
    with pytest.raises(ValueError, match="align"):
        data.SentenceDataset([row], TinyTokenizer())


def test_checkpoint_roundtrip_and_predictions(tmp_path, monkeypatch):
    model = detector().eval()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad))
    path = tmp_path / "best.pt"
    checkpoints.save_checkpoint(path, model, optimizer, 1, 17)
    monkeypatch.setattr(model1.Detector, "from_pretrained",
                        classmethod(lambda cls, config, **kwargs: cls(copy.deepcopy(model.encoder), config)))
    loaded = checkpoints.load_checkpoint(path)
    assert predict(model, TinyTokenizer(), records()[0]["text"]) == predict(loaded, TinyTokenizer(), records()[0]["text"])
    checkpoints.save_checkpoint(path, model, optimizer, 1, 17, synthetic=True)
    with pytest.raises(ValueError, match="Synthetic"):
        checkpoints.load_checkpoint(path)


def test_prompt_identity_and_no_answer_leak():
    assert model2.PROMPT_VERSION == 1
    assert hashlib.sha256(model2.PROMPT.encode()).hexdigest() == "b0bd03acba8674f79d272ca811bb20a8da6d40c74497c25de90afcac2041f6c3"
    question = model2.prompt(records()[0]["text"], "past_tense_not_inflected", [12, 18])
    assert "⟦polish⟧" in question and "polished" not in question
    assert set(model2.CATEGORIES) == set(model1.CATEGORIES)


def test_model2_notebook_uses_answer_only_loss_and_same_prompt():
    tree = notebook_definitions("02_model2_finetuning.ipynb")
    prompt_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "PROMPT" for t in n.targets))
    assert ast.literal_eval(prompt_node.value) == model2.PROMPT
    config = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "SFTConfig")
    assert any(k.arg == "completion_only_loss" and ast.literal_eval(k.value) is True for k in config.keywords)


@pytest.mark.parametrize("raw", ["not JSON", "{}", '```json\n{}\n```',
                                '{"explanation_en":"One","explanation_en":"Two"}', "NaN"])
def test_generation_parser_rejects_invalid_json(raw):
    assert validation.parse(raw)[0] is None


def test_service_skips_model2_for_negative_or_invalid_predictions():
    generator = FakeGenerator()
    diagnosis = service.DiagnosisService(FakePredictor(), generator)
    assert diagnosis.diagnose("The lantern shines.")["status"] == "no_target_error"
    diagnosis.predictor = FakePredictor.positive("unsupported", [0, 0])
    assert diagnosis.diagnose("The lantern shines.")["status"] == "unsupported_prediction"
    assert not generator.calls


def test_service_checks_edits_and_fails_without_fabricating_corrections():
    text = records()[0]["text"]
    diagnosis = service.DiagnosisService(FakePredictor.positive("past_tense_not_inflected", [12, 18]),
                                         FakeGenerator(replacement="polished"))
    result = diagnosis.diagnose(text)
    assert result["status"] == "ok" and result["corrected_text"] == "Yesterday I polished the lantern."
    diagnosis.generator = FakeGenerator(result="invalid JSON")
    failed = diagnosis.diagnose(text)
    assert failed["status"] == "generation_failed"
    assert failed["corrected_text"] is None and failed["edit"] is None


def test_model2_adapter_must_match_model_and_prompt():
    record = {"model": model2.MODEL, "model_revision": model2.MODEL_REVISION,
              "prompt_version": model2.PROMPT_VERSION, "prompt_sha256": "expected"}
    record["adapter"] = {"trained": {**record, "prompt_sha256": "different"}}
    assert model2.adapter_problems(record) == ["adapter was trained with another prompt"]


def test_model1_notebook_runs_in_order_with_test_encoder(tmp_path, monkeypatch):
    """Run the full notebook with invented records and a small fixed encoder."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = TinyTokenizer()
    tokenizer.save_pretrained = lambda path: Path(path).mkdir(parents=True)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: TinyEncoder())
    train_file, validation_file = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    for path, split, text in [(train_file, "train", records()[0]["text"]),
                              (validation_file, "validation", "Yesterday I polish a lantern.")]:
        row = {"record_schema_version": 1, "record_id": str(uuid.uuid4()), "text": text,
               "has_error": True, "category": "past_tense_not_inflected", "span": [12, 18],
               "source": {"dataset": "efcamdat", "split": split}}
        path.write_text(json.dumps(row) + "\n")
    runs = []
    for name, folder in [("01_model1_training.ipynb", "short"),
                         ("details/model1_training_details.ipynb", "full")]:
        notebook = json.loads((ROOT / name).read_text())
        output = tmp_path / folder
        scope = {"__name__": "__main__"}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), name, "exec"), scope)
            if "TRAIN_FILE" in scope and "train_records" not in scope:
                scope.update(TRAIN_FILE=train_file, VALIDATION_FILE=validation_file,
                             OUTPUT=output, EPOCHS=2, BATCH_SIZE=1)
        assert (output / "best.pt").is_file() and (output / "tokenizer").is_dir()
        history = json.loads((output / "history.json").read_text())
        assert len(history) == 2 and history[0]["validation"]["examples"] == 1
        runs.append((history, torch.load(output / "best.pt", weights_only=True)))
    short_history, short_checkpoint = runs[0]
    full_history, full_checkpoint = runs[1]
    assert short_checkpoint["epoch"] == full_checkpoint["epoch"]
    for short_row, full_row in zip(short_history, full_history):
        assert short_row["train"]["loss"] == full_row["train"]["loss"]
        assert short_row["validation"] == full_row["validation"]
    for name, weight in short_checkpoint["head_state"].items():
        assert torch.equal(weight, full_checkpoint["head_state"][name])


def test_model2_notebook_data_and_save_flow_with_mock_trainer(tmp_path, monkeypatch):
    """Check notebook wiring. This test does not run GPU fine-tuning."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    row = json.loads((ROOT / "examples/model2-format.json").read_text())
    row.update(triplet_version=2, record_id=str(uuid.uuid4()),
               source={"dataset": "efcamdat", "split": "train"},
               corrected_text="Yesterday I polished the lantern.", teacher_corrected_text=None)
    train_file = tmp_path / "triplets.jsonl"
    train_file.write_text(json.dumps(row) + "\n")
    calls = {}

    class MockTrainer:
        def __init__(self, **kwargs):
            calls.update(kwargs)
            self.state = SimpleNamespace(log_history=[{"step": 1, "loss": 0.5}])

        def train(self):
            return SimpleNamespace(training_loss=0.5)

        def save_model(self, path):
            Path(path).mkdir(parents=True)

    loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace())
    scope = {"__name__": "__main__", "Dataset": SimpleNamespace(from_list=lambda rows: rows),
             "AutoTokenizer": loader, "AutoModelForCausalLM": loader,
             "LoraConfig": lambda **kwargs: kwargs, "BitsAndBytesConfig": lambda **kwargs: kwargs,
             "SFTConfig": lambda **kwargs: kwargs, "SFTTrainer": MockTrainer}
    notebook = json.loads((ROOT / "02_model2_finetuning.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom)
                     and node.module in {"datasets", "peft", "transformers", "trl"})]
        exec(compile(tree, "model2 notebook", "exec"), scope)
        if "TRAIN_FILE" in scope and "triplets" not in scope:
            scope.update(TRAIN_FILE=train_file, OUTPUT=tmp_path / "adapter")
    run = json.loads((tmp_path / "adapter" / "run.json").read_text())
    assert run["train_triplets"] == 1 and run["prompt_version"] == model2.PROMPT_VERSION
    assert run["prompt_sha256"] == hashlib.sha256(model2.PROMPT.encode()).hexdigest()
    assert calls["args"]["completion_only_loss"] is True
    assert calls["peft_config"]["target_modules"] == model2.LORA_TARGETS


def test_short_notebook_loader_rejects_train_validation_overlap(tmp_path):
    from models.training import training_loaders

    paths = []
    for split in ["train", "validation"]:
        row = {"record_schema_version": 1, "record_id": str(uuid.uuid4()),
               "text": "The lantern shines.", "has_error": False, "category": None, "span": None,
               "source": {"dataset": "efcamdat", "split": split}}
        path = tmp_path / (split + ".jsonl")
        path.write_text(json.dumps(row) + "\n")
        paths.append(path)
    with pytest.raises(ValueError, match="overlapping"):
        training_loaders(*paths, TinyTokenizer(), batch_size=1, max_length=256, seed=42)
