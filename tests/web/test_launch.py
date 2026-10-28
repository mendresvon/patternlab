"""Launch-only flags cannot be enabled by a learner request."""

import pytest

from web.__main__ import main, parser


def test_reviewer_is_off_by_default():
    assert not parser().parse_args([]).reviewer
    assert parser().parse_args(["--reviewer"]).reviewer


@pytest.mark.parametrize("arguments", [["--checkpoint", "one.pt"], ["--adapter", "adapter"],
    ["--threshold", "nan"], ["--threshold", "2"],
    ["--background-model-load"], ["--background-model-load", "--sample"],
    ["--sample", "--checkpoint", "one.pt", "--adapter", "adapter"]])
def test_invalid_launch_options_stop_before_loading_models(arguments):
    with pytest.raises(SystemExit):
        main(arguments)


def test_launch_passes_options_to_app_without_running_a_server(monkeypatch):
    from web.app import app
    import uvicorn

    seen = []
    for key in ("mode", "reviewer", "background_model_load", "bank_path", "service_options"):
        monkeypatch.setattr(app.state, key, getattr(app.state, key))
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: seen.append((application, kwargs)))
    main(["--reviewer", "--sentence-bank", "private/bank.jsonl", "--checkpoint", "one.pt",
          "--adapter", "adapter", "--threshold", "0.3", "--port", "8010", "--background-model-load"])
    assert app.state.reviewer and app.state.mode == "live"
    assert app.state.background_model_load
    assert app.state.service_options["threshold"] == 0.3
    assert app.state.service_options["local_files_only"] is True
    assert app.state.bank_path == "private/bank.jsonl"
    assert seen[0][1] == {"host": "127.0.0.1", "port": 8010, "access_log": False}
