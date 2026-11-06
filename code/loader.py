"""Memory-efficient streaming loader.

Reads the 50k-row parquet in small slices via pyarrow ``iter_batches``
(batch_size <= 5) and materializes **one row at a time** — never a whole
batch, row group, or table. Each row's JSON columns are parsed and the
extraction is run immediately; only compact ``CallRecord`` objects (message
previews are truncated by ``Limits``) are buffered per ``trace_id`` until the
trace is complete. Raw JSON strings and parsed message trees are released as
soon as the row has been processed, so peak memory is bounded by a single row
plus the incomplete traces' compact records.

JSON failures are handled explicitly, never silently:
  * malformed JSON / wrong JSON shape (non-array messages) and documents so
    large that ``json.loads`` raises ``MemoryError`` are converted to
    :class:`JsonParseFailure`;
  * each failure is recorded with the record's ``trace_id``, ``call_index``,
    field name, and the parsing error (returned to the caller via the
    ``parse_errors`` list and printed to stderr);
  * the affected message list is left empty — no content is ever fabricated
    in place of a failed extraction, and all metadata (trace_id, call_index,
    span ids, model, timestamps) plus the source-reported / estimated /
    unknown token usage are preserved as-is.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow.parquet as pq

from config import Limits
from extractor import extract_call
from schemas import BasisTokenTotals, CallRecord, CallUsage, TraceRecord


class JsonParseFailure(Exception):
    """A JSON column could not be parsed (malformed, wrong shape, or too large)."""


def _parse_json(value: Optional[str]) -> Any:
    """Parse a JSON string, raising :class:`JsonParseFailure` instead of
    crashing the pipeline on malformed input or ``MemoryError`` from very
    large documents."""
    if not value:
        return None
    try:
        return json.loads(value)
    except MemoryError:
        # Explicitly handled: the raw string is too large to deserialize in
        # one piece. Fail only this record; the caller records trace_id and
        # the error, and streaming continues with the next row.
        raise JsonParseFailure(
            f"MemoryError in json.loads (raw field length: {len(value):,} chars)"
        )
    except RecursionError:
        raise JsonParseFailure("JSON nested beyond recursion limit")
    except (ValueError, TypeError) as exc:
        raise JsonParseFailure(f"malformed JSON: {exc}")


class TraceAccumulator:
    """Buffers compact per-call records; flushes each trace as soon as all of
    its calls (``session_call_count``) have arrived."""

    def __init__(self, limits: Limits, parse_errors: Optional[List[Dict[str, Any]]] = None):
        self.limits = limits
        self._buffers: Dict[str, Dict[str, Any]] = {}
        self.row_groups = 0
        self.parse_errors: List[Dict[str, Any]] = (
            parse_errors if parse_errors is not None else []
        )

    def _record_parse_error(self, trace_id: str, call_index: Any, field: str, failure: JsonParseFailure) -> None:
        self.parse_errors.append(
            {
                "trace_id": trace_id,
                "call_index": call_index,
                "field": field,
                "error": str(failure),
            }
        )
        print(
            f"[loader] unparseable {field} (trace_id={trace_id!r}, call_index={call_index}): {failure}",
            file=sys.stderr,
        )

    def _parse_messages(self, raw: Optional[str], field: str, trace_id: str, call_index: Any) -> List[Any]:
        try:
            parsed = _parse_json(raw)
        except JsonParseFailure as failure:
            self._record_parse_error(trace_id, call_index, field, failure)
            return []
        if parsed is None:
            return []
        if not isinstance(parsed, list):
            self._record_parse_error(
                trace_id,
                call_index,
                field,
                JsonParseFailure(f"expected a JSON array of messages, got {type(parsed).__name__}"),
            )
            return []
        return parsed

    def feed_batch(self, batch: Any) -> Iterator[Tuple[TraceRecord, List[CallRecord]]]:
        self.row_groups += 1
        names = batch.schema.names
        columns = [batch.column(idx) for idx in range(batch.num_columns)]
        for row_idx in range(batch.num_rows):
            # Materialize exactly one row: per-cell .as_py() keeps peak memory
            # at one row even when neighboring rows hold multi-MB JSON blobs.
            row = {name: column[row_idx].as_py() for name, column in zip(names, columns)}
            yield from self._feed_row(row)

    def _feed_row(self, row: Dict[str, Any]) -> Iterator[Tuple[TraceRecord, List[CallRecord]]]:
        # Capture scalars first so the raw row (and its raw JSON strings) can
        # be released immediately after extraction.
        trace_id = row.get("trace_id") or ""
        call_index = row.get("call_index")
        session_call_count = row.get("session_call_count")
        usage_basis = row.get("usage_basis") or "unknown"
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        span_id = row.get("span_id") or None
        parent_span_id = row.get("parent_span_id") or None
        model = row.get("model") or None
        start_time = row.get("start_time") or None
        end_time = row.get("end_time") or None

        try:
            extra = _parse_json(row.get("extra_json"))
        except JsonParseFailure as failure:
            self._record_parse_error(trace_id, call_index, "extra_json", failure)
            extra = None
        if not isinstance(extra, dict):
            extra = {}

        usage_extra: Dict[str, Any] = {}
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if isinstance(extra.get(key), int):
                usage_extra[key] = extra[key]
        te = extra.get("token_estimation")
        if isinstance(te, dict) and isinstance(te.get("token_estimation"), str):
            usage_extra["token_estimation_note"] = te["token_estimation"]
        session_usage = extra.get("source_reported_session_usage")
        if not isinstance(session_usage, dict):
            session_usage = None

        msgs_in = self._parse_messages(row.get("input_messages_json"), "input_messages_json", trace_id, call_index)
        msgs_out = self._parse_messages(row.get("output_messages_json"), "output_messages_json", trace_id, call_index)
        live_row = {"input_messages": msgs_in, "output_messages": msgs_out, "extra": extra}
        extraction = extract_call(live_row, self.limits)
        # Release the raw JSON strings, the parsed message trees, and the
        # parsed extra payload right away; only the compact extraction (with
        # truncated previews) survives.
        del live_row, msgs_in, msgs_out, extra, row

        call_record = CallRecord(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            call_index=call_index if call_index is not None else -1,
            session_call_count=session_call_count,
            model=model,
            start_time=start_time,
            end_time=end_time,
            usage=CallUsage(
                basis=usage_basis,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=usage_extra.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage_extra.get("cache_creation_input_tokens"),
                token_estimation_note=usage_extra.get("token_estimation_note"),
            ),
            extraction=extraction,
        )

        buf = self._buffers.get(trace_id)
        if buf is None:
            buf = {"calls": [], "expected": session_call_count, "session_usage": None}
            self._buffers[trace_id] = buf
        if buf["expected"] is None and session_call_count is not None:
            buf["expected"] = session_call_count
        buf["calls"].append({"record": call_record, "raw_call_index": call_index})
        if buf["session_usage"] is None and session_usage is not None:
            buf["session_usage"] = session_usage
        if buf["expected"] is not None and len(buf["calls"]) >= buf["expected"]:
            yield self._flush(trace_id, buf, complete=True)
            del self._buffers[trace_id]

    def flush_remaining(self) -> Iterator[Tuple[TraceRecord, List[CallRecord]]]:
        for trace_id, buf in list(self._buffers.items()):
            yield self._flush(trace_id, buf, complete=False)
            del self._buffers[trace_id]

    def _flush(self, trace_id: str, buf: Dict[str, Any], complete: bool) -> Tuple[TraceRecord, List[CallRecord]]:
        entries = sorted(buf["calls"], key=lambda e: e["raw_call_index"] if e["raw_call_index"] is not None else 0)
        call_records: List[CallRecord] = [e["record"] for e in entries]

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

        for entry in entries:
            call = entry["record"]
            usage = call.usage

            model = call.model or "unknown"
            models[model] = models.get(model, 0) + 1
            basis = usage.basis
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
            tool_call_count += len(call.extraction.tool_calls)
            for tc in call.extraction.tool_calls:
                if tc.name:
                    tools.add(tc.name)
            if first_user is None:
                for um in call.extraction.user_messages:
                    if not um.from_output:
                        first_user = um.text
                        break
            for um in call.extraction.assistant_messages:
                if um.from_output:
                    last_assistant = um.text
            if call.start_time:
                starts.append(call.start_time)
            if call.end_time:
                ends.append(call.end_time)
            if entry["raw_call_index"] is not None:
                indices.append(entry["raw_call_index"])

        expected = buf["expected"]
        trace_record = TraceRecord(
            trace_id=trace_id,
            calls=len(call_records),
            expected_calls=expected,
            complete=complete and (expected is None or len(call_records) == expected),
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

MAX_BATCH_SIZE = 5


def iter_traces(
    path: Path,
    limits: Limits,
    batch_size: int = MAX_BATCH_SIZE,
    parse_errors: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Tuple[TraceRecord, List[CallRecord]]]:
    """Stream ``(TraceRecord, [CallRecord, ...])`` pairs from the parquet file.

    * The parquet table is never loaded as a whole: batches are read via
      ``iter_batches`` with ``batch_size`` capped at 5 and rows are
      materialized one at a time.
    * ``parse_errors`` (optional list) receives one dict per unparseable JSON
      field with ``trace_id``, ``call_index``, ``field``, and the ``error``.
    """
    if batch_size > MAX_BATCH_SIZE:
        batch_size = MAX_BATCH_SIZE
    parquet_file = pq.ParquetFile(path)
    accumulator = TraceAccumulator(limits, parse_errors=parse_errors)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=_COLUMNS):
        yield from accumulator.feed_batch(batch)
    yield from accumulator.flush_remaining()
