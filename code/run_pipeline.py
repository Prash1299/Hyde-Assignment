"""Trace-mining pipeline orchestrator.

Usage:
  python code/run_pipeline.py                     # full pipeline, no LLM calls
  python code/run_pipeline.py --classify --max-batches 5
  python code/run_pipeline.py --zip               # also build submission ZIP
  python code/run_pipeline.py --limit 200         # debug on first N calls
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analytics import AnalyticsAccumulator
from batching import BatchBuilder
from config import (
    ANALYTICS_PATH,
    BATCHES_PATH,
    CLASSIFICATIONS_PATH,
    DATASET_PATH,
    LIMITS_DEFAULTS,
    OUTPUT_DIR,
    TRACE_CALLS_PATH,
    TRACE_SUMMARY_PATH,
    VALIDATION_REPORT_PATH,
    load_config,
)
from loader import iter_traces
from llm_client import HydeClassifier
from schemas import ClassificationBatch, ClassificationRecord, TraceDigest
from validate import validate_outputs
from zip_build import build_submission_zip


def run(classify: bool, max_batches: int | None, limit: int | None, build_zip: bool) -> int:
    cfg = load_config(require_api_key=classify)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"dataset not found: {DATASET_PATH}")

    generated_at = datetime.now(timezone.utc).isoformat()
    limits = LIMITS_DEFAULTS

    analytics_acc = AnalyticsAccumulator(str(DATASET_PATH.relative_to(Path(__file__).resolve().parent.parent)))
    batch_builder = BatchBuilder(model=cfg.model or "unconfigured", limits=limits)

    trace_summaries: list[dict] = []
    total_calls = 0
    stopped = limit is not None

    with TRACE_CALLS_PATH.open("w", encoding="utf-8") as calls_fh, BATCHES_PATH.open("w", encoding="utf-8") as batches_fh:
        for trace, call_records in iter_traces(DATASET_PATH, limits):
            for call in call_records:
                calls_fh.write(call.model_dump_json() + "\n")
                total_calls += 1

            analytics_acc.add_trace(trace)
            for call in call_records:
                ex = call.extraction
                analytics_acc.add_call_flags(
                    has_user=bool(ex.user_messages),
                    has_assistant=bool(ex.assistant_messages),
                    has_tool_calls=bool(ex.tool_calls),
                    has_tool_results=bool(ex.tool_results),
                    malformed=ex.malformed_message_count,
                )

            trace_summaries.append(trace.model_dump())

            digest = trace_to_digest(trace)
            for batch in batch_builder.add(digest):
                batches_fh.write(batch.model_dump_json() + "\n")

            if limit is not None and total_calls >= limit:
                stopped = True
                break

        for batch in batch_builder.finish():
            batches_fh.write(batch.model_dump_json() + "\n")

    if stopped:
        print(f"[limit] stopped after {total_calls} calls (debug mode)")

    classification_summary = {"executed": False}
    if classify:
        classification_summary = run_classification(max_batches)
    elif CLASSIFICATIONS_PATH.exists():
        classification_summary = {
            "executed": False,
            "note": "classifications.jsonl on disk is a record of a previous live attempt; not produced by this run",
        }

    TRACE_SUMMARY_PATH.write_text(json.dumps(trace_summaries, indent=1, ensure_ascii=False), encoding="utf-8")

    analytics = analytics_acc.finalize(
        row_groups=analytics_row_groups(DATASET_PATH),
        total_calls=total_calls,
        generated_at=generated_at,
        classification_summary=classification_summary,
    )
    ANALYTICS_PATH.write_text(analytics.model_dump_json(indent=1), encoding="utf-8")

    report = validate_outputs(OUTPUT_DIR, classifications_ran=classify)
    VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    print_pipeline_summary(analytics.model_dump(), report, stopped)

    if build_zip:
        zip_path, included, violations = build_submission_zip()
        print(f"[zip] wrote {zip_path.name} ({len(included)} files, violations: {len(violations)})")

    return 0 if report["all_valid"] else 1


def trace_to_digest(trace) -> TraceDigest:
    return TraceDigest(
        trace_id=trace.trace_id,
        calls=trace.calls,
        models=list(trace.models.keys()),
        tool_names=trace.distinct_tools[:25],
        first_user_message=(trace.first_user_message_preview or "")[:800] or None,
        final_assistant_message=(trace.final_assistant_message_preview or "")[:600] or None,
        tokens_by_basis={k: v.model_dump() for k, v in trace.tokens_by_basis.items()},
    )


def run_classification(max_batches: int | None) -> dict:
    cfg = load_config(require_api_key=True)
    classifier = HydeClassifier(cfg)

    batches = [ClassificationBatch.model_validate(json.loads(line)) for line in BATCHES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_batches is not None:
        batches = batches[:max_batches]

    ran = 0
    ok = 0
    failed = 0
    with CLASSIFICATIONS_PATH.open("w", encoding="utf-8") as fh:
        for batch in batches:
            records: list[ClassificationRecord] = classifier.classify_batch(batch)
            for record in records:
                fh.write(record.model_dump_json() + "\n")
                if record.validated:
                    ok += 1
                else:
                    failed += 1
            ran += 1
            print(f"[classify] {batch.batch_id}: {len(records)} records ({ok} valid so far)")
    return {
        "executed": True,
        "model": cfg.model,
        "batches_processed": ran,
        "valid_classifications": ok,
        "invalid_or_failed": failed,
        "note": "all classifications come verbatim from Hyde GLM-5.3-Flash API responses; none are synthesized",
    }


def analytics_row_groups(dataset_path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(dataset_path).num_row_groups


def print_pipeline_summary(analytics: dict, report: dict, stopped: bool) -> None:
    print("\n=== PIPELINE SUMMARY ===")
    print(f"calls: {analytics['total_calls']}  traces: {analytics['total_traces']}  (complete: {analytics['complete_traces']})")
    print(f"tokens_by_basis: {json.dumps({k: {'calls': v['calls'], 'in': v['input_tokens'], 'out': v['output_tokens']} for k, v in analytics['tokens_by_basis'].items()}, indent=1)}")
    print(f"tool_calls: {analytics['tool_call_count']}  distinct_tools: {len(analytics['distinct_tools'])}")
    print(f"classification: {json.dumps(analytics['classification'])}")
    print(f"validation all_valid: {report['all_valid']}")
    for name, info in report["files"].items():
        print(f"  {name}: {info['records']} records, status={info['status']}")
    if stopped:
        print("NOTE: debug limit was active; outputs are partial.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace-mining pipeline (Hyde assignment)")
    parser.add_argument("--classify", action="store_true", help="run Hyde GLM-5.3-Flash classification (makes real API calls)")
    parser.add_argument("--max-batches", type=int, default=None, help="limit number of batches classified")
    parser.add_argument("--limit", type=int, default=None, help="debug: stop after N calls")
    parser.add_argument("--zip", action="store_true", help="build submission ZIP after the pipeline")
    args = parser.parse_args()
    return run(classify=args.classify, max_batches=args.max_batches, limit=args.limit, build_zip=args.zip)


if __name__ == "__main__":
    raise SystemExit(main())
