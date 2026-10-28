"""Skip the web suite when the optional `web` extra is not installed."""

import importlib.util

# The research-only command (`--extra dev --extra synth`) has no FastAPI. Every web
# test module imports the app, so collect none of them rather than fail collection.
collect_ignore_glob = [] if importlib.util.find_spec("fastapi") else ["test_*.py"]
