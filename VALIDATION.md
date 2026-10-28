# Validation record

Checked on October 28, 2026, against research source revision
`7c811efa8b5e9889f426c7aa4de99721249a5d59`.

## Completed checks

| Check | Result |
| --- | --- |
| Reference model and website suite | 308 tests passed before extraction |
| Standalone model, notebook, and website suite | 217 tests passed |
| Executable model definitions | 49 extracted definitions match the reference |
| Small-model numerical comparison | Identical aligned records, batches, scores, losses, and gradients |
| Existing Model 1 checkpoint | Identical predictions on eight invented sentences |
| Model 2 prompts | Identical prompts for all seven categories |
| Diagnosis service | Matching responses for valid and invalid generated outputs |
| Website files | Code preserved with local model imports. Templates, question bank, and assets preserved |
| Model 1 notebook execution | All cells ran in order with invented records and a small test encoder |
| Short Model 1 walkthrough | 59 code lines. Two test epochs match the full notebook's losses, validation metrics, saved weights, and best-checkpoint choice |
| Model 2 notebook wiring | Data validation, trainer settings, and saving checked with a mock trainer |
| Website-only environment | Startup and sample requests passed with no PyTorch installed |
| Public-file scan | No private datasets, weights, credentials, machine paths, or executed notebook outputs found |

The Model 1 numerical check uses fixed test weights. The existing checkpoint check uses only invented
sentences. Neither check copies learner records or model weights into this repository.

The short Model 1 notebook keeps the forward pass, loss, gradient calculation, gradient clipping, and
optimizer update visible. Token alignment, batch preparation, and validation metrics use supporting files.
The full notebook remains under `details/`. The short version logs training loss only. The full version
also logs training classification metrics. Both versions retain the original validation metrics.

The exact Model 2 prompt version is 1. Its SHA-256 hash is:

```text
b0bd03acba8674f79d272ca811bb20a8da6d40c74497c25de90afcac2041f6c3
```

## Limits

These checks establish software consistency for the paths tested. They do not establish new model
accuracy results. Full GPU training, Gemma generation, and the live two-model website were not run
on this machine. The Model 2 mock test checks notebook wiring only.

Retraining can produce different weights and scores. New runs need their own held-out evaluation.
The website files retain their existing interface. A new visual browser review was not performed.
Website tests use an in-process HTTP client. No public server was started.

FastAPI's test client reports one upstream deprecation warning about HTTPX. The checks pass with the
listed dependency versions.
