"""Post-write validation pass: re-reads every emitted artifact from disk and
validates each record against the pydantic schemas."""

import json
from pathlib import Path
from typing import Any, Dict, List

from schemas import (
    Analytics,
    CallRecord,
    ClassificationBatch,
    ClassificationRecord,
    TraceRecord,
)

MAX_EXAMPLES = 20


def _validate_jsonl(path: Path, model: type) -> Dict[str, Any]:
    result: Dict[str, Any] = {"file": path.name, "exists": path.exists(), "records": 0, "validation_errors": 0, "examples": []}
    if not path.exists():
        result["status"] = "missing"
        return result
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            result["records"] += 1
            try:
                model.model_validate(json.loads(line))
            except Exception as exc:
                result["validation_errors"] += 1
                if len(result["examples"]) < MAX_EXAMPLES:
                    result["examples"].append({"line": line_no, "error": str(exc)[:300]})
    result["status"] = "valid" if result["validation_errors"] == 0 else "invalid"
    return result


def _validate_json_file(path: Path, model: type) -> Dict[str, Any]:
    result: Dict[str, Any] = {"file": path.name, "exists": path.exists(), "records": 0, "validation_errors": 0, "examples": []}
    if not path.exists():
        result["status"] = "missing"
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["status"] = "invalid"
        result["examples"].append({"error": f"json parse failed: {str(exc)[:300]}"})
        return result
    items = data if isinstance(data, list) else [data]
    for idx, item in enumerate(items):
        result["records"] += 1
        try:
            model.model_validate(item)
        except Exception as exc:
            result["validation_errors"] += 1
            if len(result["examples"]) < MAX_EXAMPLES:
                result["examples"].append({"index": idx, "error": str(exc)[:300]})
    result["status"] = "valid" if result["validation_errors"] == 0 else "invalid"
    return result


def validate_outputs(output_dir: Path, classifications_ran: bool) -> Dict[str, Any]:
    files: Dict[str, Any] = {
        "trace_calls.jsonl": _validate_jsonl(output_dir / "trace_calls.jsonl", CallRecord),
        "trace_summary.json": _validate_json_file(output_dir / "trace_summary.json", TraceRecord),
        "classification_batches.jsonl": _validate_jsonl(output_dir / "classification_batches.jsonl", ClassificationBatch),
        "analytics.json": _validate_json_file(output_dir / "analytics.json", Analytics),
    }
    if classifications_ran:
        files["classifications.jsonl"] = _validate_jsonl(output_dir / "classifications.jsonl", ClassificationRecord)

    all_valid = all(v.get("status") == "valid" for v in files.values())
    return {
        "validator": "pydantic v2",
        "all_valid": all_valid,
        "classification_stage_executed": classifications_ran,
        "files": files,
    }
