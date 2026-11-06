"""Builds the submission ZIP. Explicitly excludes the dataset, virtualenv,
.env (API keys), caches, and any file containing a key-shaped secret."""

import os
import zipfile
from pathlib import Path
from typing import List, Tuple

from config import PROJECT_ROOT, SUBMISSION_ZIP_PATH

EXCLUDED_DIRS = {"dataset", ".venv", "venv", "__pycache__", ".git", ".idea", ".vscode", "output_previous"}
EXCLUDED_FILES = {".env", ".env.local", "traces.parquet", "inspect_dataset.py"}
EXCLUDED_EXTENSIONS = {".pyc", ".pyo", ".parquet"}
SECRET_MARKERS = ("sk-", "HYDE_API_KEY=")


def _iter_included_files() -> List[Path]:
    included: List[Path] = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for name in files:
            path = Path(root) / name
            rel = path.relative_to(PROJECT_ROOT)
            if name in EXCLUDED_FILES or path.suffix.lower() in EXCLUDED_EXTENSIONS:
                continue
            if str(rel).startswith(".venv"):
                continue
            included.append(path)
    return sorted(included)


def _scan_for_secrets(files: List[Path]) -> List[str]:
    flagged: List[str] = []
    env_key = os.getenv("HYDE_API_KEY", "")
    for path in files:
        try:
            if path.suffix.lower() in {".zip", ".png", ".jpg", ".jpeg", ".pdf"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if env_key and env_key in text:
            flagged.append(f"{path}: contains the live HYDE_API_KEY value")
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for marker in SECRET_MARKERS:
                if marker in stripped and ("API_KEY" in stripped.upper() or marker == "sk-"):
                    if marker == "sk-" and "sk-" in stripped.lower() and "API" not in stripped.upper() and "KEY" not in stripped.upper():
                        continue
                    flagged.append(f"{path}: potential secret pattern {marker!r}")
                    break
    return flagged


def build_submission_zip() -> Tuple[Path, List[str], List[str]]:
    files = _iter_included_files()
    flagged = _scan_for_secrets(files)
    if flagged:
        raise RuntimeError("Refusing to build ZIP, potential secrets found:\n" + "\n".join(flagged))

    if SUBMISSION_ZIP_PATH.exists():
        SUBMISSION_ZIP_PATH.unlink()

    with zipfile.ZipFile(SUBMISSION_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(PROJECT_ROOT))

    with zipfile.ZipFile(SUBMISSION_ZIP_PATH, "r") as zf:
        names = zf.namelist()

    violations = []
    for name in names:
        lowered = name.lower()
        if "dataset" in lowered or lowered.endswith(".parquet") or lowered == ".env" or ".env" in lowered.split("/")[-1]:
            violations.append(name)
    if violations:
        raise RuntimeError("ZIP contains excluded paths: " + ", ".join(violations))

    return SUBMISSION_ZIP_PATH, files, violations
