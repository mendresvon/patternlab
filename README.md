# PatternLab

**English Grammar Diagnosis and Practice for Mandarin Speakers**

PatternLab is the presentation version of a student research project. It detects one target grammar error,
explains it in Traditional Chinese and English, and suggests one correction. The website also provides
lessons, practice questions, quizzes, and progress tracking.

This repository contains the code for the oral defense (專題口試). It has its own Git history.
It does not need access to the private research repository. Private learner records and trained model
files are not included.

## Start with the code walkthrough

| File | What to explain |
| --- | --- |
| `01_model1_training.ipynb` | Load prepared records, align labels, define the detector, train, validate, and save |
| `02_model2_finetuning.ipynb` | Build prompts, configure QLoRA, fine-tune Gemma, and save the adapter |
| `03_running_the_models.ipynb` | Load both models, inspect the diagnosis, generate an edit, and validate it |
| `models/` | Shared network definitions, input checks, checkpoint loading, and inference |
| `web/` | The full FastAPI, Jinja2, and JavaScript website |
| `tests/` | Software checks with invented examples |
| `examples/` | Invented examples that show the data fields |

The training steps are visible in the notebooks. The website uses the small shared Python modules.
Tests check that repeated notebook definitions match those modules. Open Jupyter from this folder and
run each notebook from top to bottom. Each training notebook needs prepared private data before it can run.

## Try the website without model files

Use Python 3.11 or later. In this folder, run:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m web --sample
```

Open [PatternLab locally](http://127.0.0.1:8000).

Sample mode displays clearly labeled, prewritten examples. It does not run either model. Entering other
text in this mode does not produce a model diagnosis. Lessons, all seven practice topics, quizzes,
progress, and XP work without model files. Running `python -m web` without model files keeps practice
available and reports that live diagnosis is unavailable.

The website retains the original home page, lessons, multiple-choice and typed questions, five-question
rounds, answer feedback, progress, accuracy, XP, progress reset, mobile navigation, sentence selector,
error highlights, insertion markers, correction copying, sample mode, and reviewer mode.
Progress stays in the current browser. Writing Lab does not save submitted sentences.

## Train or run the models

Install the model and notebook dependencies in an environment suitable for your GPU:

```sh
python -m pip install -r requirements-models.txt
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

The model requirements retain the research versions for the main model libraries. GPU training and
4-bit Gemma inference need a compatible NVIDIA CUDA environment. Keep a provider's matching PyTorch
and driver setup when necessary. `bitsandbytes` is installed on Linux only. The notebooks do not assume
a private GitHub account or clone a private repository.

Model 1 uses fixed RoBERTa token representations and a trainable bidirectional LSTM with attention.
It predicts error presence, one category, and start/end boundaries. Model 2 uses Gemma with quantized
low-rank adaptation (QLoRA). QLoRA trains small added matrices while the base model weights stay fixed.
Its training loss applies to answer tokens only.

Set the paths in the notebook settings cells. The suggested private layout is outside this repository:

```text
~/patternlab-private/
  model1/train.jsonl
  model1/validation.jsonl
  model2/train.jsonl
  runs/model1-presentation/best.pt
  runs/model1-presentation/tokenizer/
  runs/model2-presentation/adapter_config.json
  runs/model2-presentation/adapter_model.safetensors
  runs/model2-presentation/run.json
  bank.jsonl
```

The JSON files in `examples/` show the main fields only. They are invented format examples.
They are not approved training data and do not claim corpus provenance. Model 1 training uses the
existing version 1 record format. Model 2 training uses the existing version 2 triplet format.
The shared validators describe the required metadata. Dataset extraction, annotation, AI review,
and research evaluation are outside this presentation repository.

Model downloads require the applicable base-model access. Complete that access in your own environment.
The live website loads cached base models and does not download private checkpoints. Existing compatible
research checkpoints and adapters can also be supplied. Use the validation threshold for the checkpoint
you load. The value below is a placeholder, not a tuned result:

```sh
python -m web \
  --checkpoint "$HOME/patternlab-private/runs/model1-presentation/best.pt" \
  --adapter "$HOME/patternlab-private/runs/model2-presentation" \
  --device cuda \
  --threshold 0.5
```

Add `--reviewer` to show English beside Chinese. Add `--background-model-load` to serve pages while the
models load. Add `--sentence-bank "$HOME/patternlab-private/bank.jsonl"` to enable the private demonstration
sentence selector. Its current format requires 85 unique short sentences: ten per category and fifteen
clean sentences. Only the sentence text reaches the browser. The bank is not included here.

`/health` reports website, practice, and model readiness. Missing model files do not trigger a sample
fallback. Invalid predictions and invalid generated edits do not produce a speculative correction.

## Limits and results

The seven target categories are article omission, plural marking omission, past tense not inflected,
subject-verb agreement, be-verb omission, to omission, and degree-adverb misuse.
The system reports at most one target error per sentence. A negative result does not prove that a
sentence has no other errors. A predicted category does not prove that Mandarin caused the error.

The network, loss, character alignment, prompt text, model revisions, and generation checks retain their
research behavior. The prompt template keeps its original wording because existing adapters depend on it.
New training can produce different weights and scores. No performance results are claimed for a new
presentation training run. CPU software checks do not establish GPU behavior or model accuracy.

## Verify the software

Install the website and model requirements, then run:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest
python tools/check_public_files.py
```

The tests use invented examples and small test models. They do not need private data, model downloads,
or paid API calls. They check training updates, frozen weights, notebook consistency, checkpoint loading,
prompt compatibility, model response validation, and website behavior.

Before sharing notebook changes, restart the kernel and clear all outputs. The public-file check rejects
executed notebooks, common credential files, and model weights. Keep the private files outside this folder.

## Source and validation

This code was derived from the research implementation at source revision
`7c811efa8b5e9889f426c7aa4de99721249a5d59`. No research Git history was copied.
See `VALIDATION.md` for the checks performed on this presentation version and their limits.
