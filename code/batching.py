"""Prepares bounded classification batches from per-trace digests.

Batches are size-limited (trace count and serialized character budget) so
each one fits comfortably in a single Hyde GLM-5.3-Flash request. Batches
are marked status="prepared" until the LLM stage actually runs.
"""

import json
from typing import Iterator, List

from config import Limits
from schemas import ClassificationBatch, TraceDigest


class BatchBuilder:
    def __init__(self, model: str, limits: Limits):
        self.model = model
        self.limits = limits
        self._pending: List[TraceDigest] = []
        self._pending_chars = 0

    def add(self, digest: TraceDigest) -> Iterator[ClassificationBatch]:
        size = len(digest.model_dump_json())
        if self._pending and (len(self._pending) >= self.limits.max_batch_traces or self._pending_chars + size > self.limits.max_batch_chars):
            yield self._emit()
        self._pending.append(digest)
        self._pending_chars += size

    def finish(self) -> Iterator[ClassificationBatch]:
        if self._pending:
            yield self._emit()

    def _emit(self) -> ClassificationBatch:
        from datetime import datetime, timezone

        batch = ClassificationBatch(
            batch_id=f"batch_{BatchBuilder._next_id():04d}",
            model=self.model,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="prepared",
            trace_count=len(self._pending),
            traces=self._pending,
        )
        self._pending = []
        self._pending_chars = 0
        return batch

    _id_counter = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter - 1


def batch_to_payload(batch: ClassificationBatch) -> str:
    """Compact JSON payload of the batch sent to the model."""
    return json.dumps([t.model_dump() for t in batch.traces], ensure_ascii=False)


CLASSIFICATION_INSTRUCTIONS = """You are classifying LLM agent traces (a trace = one session of ordered LLM API calls).
For EACH trace in the provided JSON array, judge only from the evidence in that data.
Return ONLY a JSON array, one object per trace, with exactly these fields:
  trace_id (copy verbatim), primary_intent (short phrase), domain (one of:
  software-engineering, customer-support, shopping-assistant, data-analysis,
  browsing-research, writing, other), uses_tools (boolean), outcome
  (one of: completed, partially_completed, unclear), confidence (0.0-1.0),
  rationale (one sentence grounded in the messages).
Do not invent facts not supported by the trace. If evidence is insufficient, use outcome "unclear" and a low confidence."""
