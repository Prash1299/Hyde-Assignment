"""Hyde GLM-5.3-Flash client (OpenAI-compatible) for the classification stage.

Runs ONLY when explicitly requested (--classify). Results are stored
verbatim from real API responses; nothing is ever synthesized locally.
"""

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

from batching import CLASSIFICATION_INSTRUCTIONS, ClassificationBatch, batch_to_payload
from config import HydeConfig
from schemas import ClassificationRecord, ClassificationResultPayload


def _extract_json_array(text: str) -> Optional[Any]:
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except (ValueError, TypeError):
            pass
    bracketed = re.search(r"\[.*\]", text, re.DOTALL)
    if bracketed:
        try:
            return json.loads(bracketed.group(0))
        except (ValueError, TypeError):
            return None
    return None


def _api_usage(response: Any) -> Optional[Dict[str, Any]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    data: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            data[key] = value
    return data or None


class HydeClassifier:
    def __init__(self, cfg: HydeConfig):
        if not (cfg.api_key and cfg.base_url and cfg.model):
            raise ValueError("Hyde GLM configuration incomplete; set HYDE_API_KEY/HYDE_BASE_URL/HYDE_MODEL")
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.request_timeout_s, max_retries=0)

    def _request(self, batch: ClassificationBatch) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": CLASSIFICATION_INSTRUCTIONS},
                        {"role": "user", "content": batch_to_payload(batch)},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.cfg.max_retries:
                    time.sleep(2 ** attempt)
        raise last_error  # type: ignore[misc]

    def classify_batch(self, batch: ClassificationBatch) -> List[ClassificationRecord]:
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            response = self._request(batch)
        except Exception as exc:
            return [
                ClassificationRecord(
                    batch_id=batch.batch_id,
                    trace_id=digest.trace_id,
                    model=self.cfg.model or "unknown",
                    created_at=created_at,
                    response_raw="",
                    error=f"request_failed: {exc}",
                )
                for digest in batch.traces
            ]

        content = response.choices[0].message.content or ""
        usage = _api_usage(response)
        parsed_items = _extract_json_array(content)
        parsed_by_trace: Dict[str, Dict[str, Any]] = {}
        if isinstance(parsed_items, list):
            for item in parsed_items:
                if isinstance(item, dict) and isinstance(item.get("trace_id"), str):
                    parsed_by_trace[item["trace_id"]] = item

        records: List[ClassificationRecord] = []
        for digest in batch.traces:
            item = parsed_by_trace.get(digest.trace_id)
            record = ClassificationRecord(
                batch_id=batch.batch_id,
                trace_id=digest.trace_id,
                model=self.cfg.model or "unknown",
                created_at=created_at,
                response_raw=content,
                api_usage=usage,
            )
            if item is None:
                record.error = "trace_missing_from_model_response"
            else:
                record.parsed = item
                try:
                    ClassificationResultPayload.model_validate(item)
                    record.validated = True
                except Exception as exc:
                    record.validation_errors = [str(exc)]
            records.append(record)
        return records
