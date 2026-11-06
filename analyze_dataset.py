import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import pyarrow.parquet as pq

DATASET = Path("dataset/traces.parquet")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

parquet = pq.ParquetFile(DATASET)

# One entry per trace/session
traces = {}

# Dataset-wide statistics
models = Counter()
usage_bases = Counter()
total_input_tokens = 0
total_output_tokens = 0
known_input_calls = 0
known_output_calls = 0
missing_input_tokens = 0
missing_output_tokens = 0
total_calls = 0

def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

columns = [
    "trace_id", "call_index", "session_call_count",
    "model", "input_tokens", "output_tokens",
    "usage_basis", "start_time", "end_time"
]

for batch in parquet.iter_batches(
    batch_size=500,
    columns=columns
):
    for row in batch.to_pylist():
        total_calls += 1

        trace_id = row["trace_id"]
        model = row["model"] or "unknown"
        usage = row["usage_basis"] or "unknown"

        models[model] += 1
        usage_bases[usage] += 1

        input_tokens = row["input_tokens"]
        output_tokens = row["output_tokens"]

        if input_tokens is not None:
            total_input_tokens += input_tokens
            known_input_calls += 1
        else:
            missing_input_tokens += 1

        if output_tokens is not None:
            total_output_tokens += output_tokens
            known_output_calls += 1
        else:
            missing_output_tokens += 1

        if trace_id not in traces:
            traces[trace_id] = {
                "trace_id": trace_id,
                "calls": 0,
                "expected_calls": row["session_call_count"],
                "models": Counter(),
                "usage_bases": Counter(),
                "input_tokens": 0,
                "output_tokens": 0,
                "known_input_calls": 0,
                "known_output_calls": 0,
                "first_call_index": None,
                "last_call_index": None,
                "earliest_start": None,
                "latest_end": None,
            }

        trace = traces[trace_id]
        trace["calls"] += 1
        trace["models"][model] += 1
        trace["usage_bases"][usage] += 1

        if input_tokens is not None:
            trace["input_tokens"] += input_tokens
            trace["known_input_calls"] += 1

        if output_tokens is not None:
            trace["output_tokens"] += output_tokens
            trace["known_output_calls"] += 1

        index = row["call_index"]
        if index is not None:
            if trace["first_call_index"] is None:
                trace["first_call_index"] = index
            trace["first_call_index"] = min(
                trace["first_call_index"], index
            )
            trace["last_call_index"] = max(
                trace["last_call_index"] if trace["last_call_index"] is not None else index,
                index
            )

        start = parse_time(row["start_time"])
        end = parse_time(row["end_time"])

        if start and (
            trace["earliest_start"] is None
            or start < trace["earliest_start"]
        ):
            trace["earliest_start"] = start

        if end and (
            trace["latest_end"] is None
            or end > trace["latest_end"]
        ):
            trace["latest_end"] = end


# Prepare JSON-safe trace summaries
trace_summaries = []

for trace in traces.values():
    trace_summaries.append({
        "trace_id": trace["trace_id"],
        "calls": trace["calls"],
        "expected_calls": trace["expected_calls"],
        "models": dict(trace["models"]),
        "usage_bases": dict(trace["usage_bases"]),
        "input_tokens": trace["input_tokens"],
        "output_tokens": trace["output_tokens"],
        "known_input_calls": trace["known_input_calls"],
        "known_output_calls": trace["known_output_calls"],
        "first_call_index": trace["first_call_index"],
        "last_call_index": trace["last_call_index"],
        "earliest_start": (
            trace["earliest_start"].isoformat()
            if trace["earliest_start"] else None
        ),
        "latest_end": (
            trace["latest_end"].isoformat()
            if trace["latest_end"] else None
        ),
    })

trace_summaries.sort(key=lambda x: x["trace_id"])

analytics = {
    "dataset": str(DATASET),
    "total_calls": total_calls,
    "total_traces": len(traces),
    "row_groups": parquet.num_row_groups,
    "models": dict(models.most_common()),
    "usage_bases": dict(usage_bases.most_common()),
    "token_statistics": {
        "input_tokens_sum_known_values": total_input_tokens,
        "output_tokens_sum_known_values": total_output_tokens,
        "known_input_calls": known_input_calls,
        "known_output_calls": known_output_calls,
        "missing_input_calls": missing_input_tokens,
        "missing_output_calls": missing_output_tokens,
    },
    "traces_with_call_count_mismatch": sum(
        1 for t in trace_summaries
        if t["expected_calls"] is not None
        and t["calls"] != t["expected_calls"]
    ),
}

with open(OUTPUT / "analytics.json", "w", encoding="utf-8") as f:
    json.dump(analytics, f, indent=2)

with open(OUTPUT / "trace_summary.json", "w", encoding="utf-8") as f:
    json.dump(trace_summaries, f, indent=2)

print("ANALYSIS COMPLETE")
print("Total calls:", total_calls)
print("Total traces:", len(traces))
print("Models:", dict(models))
print("Usage bases:", dict(usage_bases))
print("Known input tokens:", total_input_tokens)
print("Known output tokens:", total_output_tokens)
print("Output directory:", OUTPUT.resolve())