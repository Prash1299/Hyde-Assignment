"""Memory-efficient streaming loader.

Reads the 50k-row parquet in batches via pyarrow ``iter_batches``, buffers
calls per trace_id, and flushes each trace as soon as all of its calls
(session_call_count) are collected — so memory is bounded by the number of
*incomplete* traces at any time, never by the full dataset.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow.parquet as pq

from config import Limits
from extractor import extract_call
from schemas import BasisTokenTotals, CallRecord, CallUsage, TraceRecord


def _parse_json(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    extra = _parse_json(row.get("extra_json"))
    usage_extra: Dict[str, Any] = {}
    if isinstance(extra, dict):
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if isinstance(extra.get(key), int):
                usage_extra[key] = extra[key]
        te = extra.get("token_estimation")
        if isinstance(te, dict) and isinstance(te.get("token_estimation"), str):
            usage_extra["token_estimation_note"] = te["token_estimation"]
    return {
        "trace_id": row.get("trace_id") or "",
        "span_id": row.get("span_id") or None,
        "parent_span_id": row.get("parent_span_id") or None,
        "call_index": row.get("call_index"),
        "session_call_count": row.get("session_call_count"),
        "model": row.get("model") or None,
        "start_time": row.get("start_time") or None,
        "end_time": row.get("end_time") or None,
        "usage_basis": row.get("usage_basis") or "unknown",
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        # Raw JSON strings are buffered unparsed; message trees are materialized
        # one call at a time in _flush to keep peak memory low.
        "input_messages_json": row.get("input_messages_json"),
        "output_messages_json": row.get("output_messages_json"),
        "extra": extra if isinstance(extra, dict) else {},
        "usage_extra": usage_extra,
    }


class TraceAccumulator:
    def __init__(self, limits: Limits):
        self.limits = limits
        self._buffers: Dict[str, Dict[str, Any]] = {}
        self.row_groups = 0

    def feed_batch(self, batch: Any) -> Iterator[Tuple[TraceRecord, List[CallRecord]]]:
        self.row_groups += 1
        for row in batch.to_pylist():
            normalized = _normalize_row(row)
            trace_id = normalized["trace_id"]
            buf = self._buffers.get(trace_id)
            if buf is None:
                buf = {"calls": [], "expected": normalized["session_call_count"], "session_usage": None}
                self._buffers[trace_id] = buf
            if buf["expected"] is None and normalized["session_call_count"] is not None:
                buf["expected"] = normalized["session_call_count"]
            buf["calls"].append(normalized)
            su = buf["session_usage"]
            if su is None and isinstance(normalized["extra"].get("source_reported_session_usage"), dict):
                buf["session_usage"] = normalized["extra"]["source_reported_session_usage"]
            if buf["expected"] is not None and len(buf["calls"]) >= buf["expected"]:
                yield self._flush(trace_id, buf, complete=True)
                del self._buffers[trace_id]

    def flush_remaining(self) -> Iterator[Tuple[TraceRecord, List[CallRecord]]]:
        for trace_id, buf in list(self._buffers.items()):
            yield self._flush(trace_id, buf, complete=False)
            del self._buffers[trace_id]

    def _flush(self, trace_id: str, buf: Dict[str, Any], complete: bool) -> Tuple[TraceRecord, List[CallRecord]]:
        calls: List[Dict[str, Any]] = sorted(buf["calls"], key=lambda c: (c["call_index"] if c["call_index"] is not None else 0))

        call_records: List[CallRecord] = []
        models: Dict[str, int] = {}
        usage_bases: Dict[str, int] = {}
        tokens_by_basis: Dict[str, BasisTokenTotals] = {}
        cache_read = 0
        cache_creation = 0
        calls_with_cache = 0
        tool_call_count = 0
        tools: set = set()
        first_user: Optional[str] = None
        last_assistant: Optional[str] = None
        starts: List[str] = []
        ends: List[str] = []
        indices: List[int] = []

        for call in calls:
            msgs_in = _parse_json(call["input_messages_json"]) or []
            msgs_out = _parse_json(call["output_messages_json"]) or []
            live_row = {"input_messages": msgs_in, "output_messages": msgs_out, "extra": call["extra"]}
            extraction = extract_call(live_row, self.limits)
            del live_row, msgs_in, msgs_out
            usage_extra = call["usage_extra"]
            usage = CallUsage(
                basis=call["usage_basis"],
                input_tokens=call["input_tokens"],
                output_tokens=call["output_tokens"],
                cache_read_input_tokens=usage_extra.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage_extra.get("cache_creation_input_tokens"),
                token_estimation_note=usage_extra.get("token_estimation_note"),
            )
            call_records.append(
                CallRecord(
                    trace_id=trace_id,
                    span_id=call["span_id"],
                    parent_span_id=call["parent_span_id"],
                    call_index=call["call_index"] if call["call_index"] is not None else -1,
                    session_call_count=call["session_call_count"],
                    model=call["model"],
                    start_time=call["start_time"],
                    end_time=call["end_time"],
                    usage=usage,
                    extraction=extraction,
                )
            )

            model = call["model"] or "unknown"
            models[model] = models.get(model, 0) + 1
            basis = call["usage_basis"]
            usage_bases[basis] = usage_bases.get(basis, 0) + 1
            bucket = tokens_by_basis.setdefault(basis, BasisTokenTotals())
            bucket.calls += 1
            if usage.input_tokens is not None:
                bucket.input_tokens += usage.input_tokens
                bucket.known_input_calls += 1
            if usage.output_tokens is not None:
                bucket.output_tokens += usage.output_tokens
                bucket.known_output_calls += 1
            if usage.cache_read_input_tokens is not None or usage.cache_creation_input_tokens is not None:
                calls_with_cache += 1
                cache_read += usage.cache_read_input_tokens or 0
                cache_creation += usage.cache_creation_input_tokens or 0
            tool_call_count += len(extraction.tool_calls)
            for tc in extraction.tool_calls:
                if tc.name:
                    tools.add(tc.name)
            if first_user is None:
                for um in extraction.user_messages:
                    if not um.from_output:
                        first_user = um.text
                        break
            for um in extraction.assistant_messages:
                if um.from_output:
                    last_assistant = um.text
            if call["start_time"]:
                starts.append(call["start_time"])
            if call["end_time"]:
                ends.append(call["end_time"])
            if call["call_index"] is not None:
                indices.append(call["call_index"])

        expected = buf["expected"]
        trace_record = TraceRecord(
            trace_id=trace_id,
            calls=len(calls),
            expected_calls=expected,
            complete=complete and (expected is None or len(calls) == expected),
            models=models,
            usage_bases=usage_bases,
            tokens_by_basis={k: v for k, v in tokens_by_basis.items()},
            cache_read_input_tokens=cache_read if calls_with_cache else None,
            cache_creation_input_tokens=cache_creation if calls_with_cache else None,
            calls_with_cache_fields=calls_with_cache,
            session_usage_source_reported=buf["session_usage"],
            tool_call_count=tool_call_count,
            distinct_tools=sorted(tools),
            first_user_message_preview=first_user,
            final_assistant_message_preview=last_assistant,
            first_call_index=min(indices) if indices else None,
            last_call_index=max(indices) if indices else None,
            earliest_start=min(starts) if starts else None,
            latest_end=max(ends) if ends else None,
            call_order_preserved=indices == sorted(indices),
        )
        return trace_record, call_records


_COLUMNS = [
    "trace_id", "span_id", "parent_span_id", "call_index", "session_call_count",
    "model", "start_time", "end_time", "usage_basis", "input_tokens", "output_tokens",
    "input_messages_json", "output_messages_json", "extra_json",
]


def iter_traces(path: Path, limits: Limits, batch_size: int = 128) -> Iterator[Tuple[TraceRecord, List[CallRecord]]]:
    parquet_file = pq.ParquetFile(path)
    accumulator = TraceAccumulator(limits)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=_COLUMNS):
        yield from accumulator.feed_batch(batch)
    yield from accumulator.flush_remaining()
