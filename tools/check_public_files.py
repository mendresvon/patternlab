"""Check tracked files before public sharing. Print file names, never file contents."""

import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".jsonl", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".xml", ".sqlite", ".sqlite3"}
FORBIDDEN_NAMES = {".env", "credentials.json", "token.json", "id_rsa", "id_ed25519"}


def check(root=ROOT):
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                            cwd=root, capture_output=True, text=True)
    if result.returncode:
        paths = [p for p in root.rglob("*") if p.is_file() and
                 not any(part.startswith(".") for part in p.relative_to(root).parts)]
    else:
        paths = [root / name for name in result.stdout.splitlines()]
    problems = []
    for path in paths:
        name = str(path.relative_to(root))
        if path.suffix in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES:
            problems.append(name + ": private file type")
        if path.is_symlink():
            problems.append(name + ": symbolic link")
            continue
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    problems.append(name + ": executed notebook")
                    break
        if path.suffix in {".py", ".md", ".json", ".ipynb", ".toml", ".txt", ".html", ".js", ".css"}:
            text = path.read_text(encoding="utf-8")
            if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)", text):
                problems.append(name + ": possible credential")
            research_package = "english" + "_l1_transfer_diagnostics"
            research_folder = "english" + "-l1-transfer-diagnostics"
            if (re.search(r"/(?:Users|home)/[A-Za-z0-9_.-]+/", text)
                    or research_package in text or research_folder in text):
                problems.append(name + ": private project dependency or machine path")
    return problems


if __name__ == "__main__":
    problems = check()
    if problems:
        print("Public-file check failed:")
        print("\n".join(problems))
        sys.exit(1)
    print("Public-file check passed.")
