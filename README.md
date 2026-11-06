# Trace-Mining Pipeline — Hyde Assessment

A memory-efficient pipeline that streams the 50,000-record LLM-trace dataset
(`dataset/traces.parquet`), groups calls into traces, extracts structured
conversation content, and prepares semantic-classification batches for the
configured **Hyde GLM-5.3-Flash** model.

**Stack:** Python 3.12+, pyarrow, pydantic v2, python-dotenv, openai (client).

## Run

```powershell
# from the project root, using the provided virtualenv
.venv\Scripts\python.exe code\run_pipeline.py                # full pipeline, no LLM calls
.venv\Scripts\python.exe code\run_pipeline.py --classify --max-batches 5   # live classification
.venv\Scripts\python.exe code\run_pipeline.py --limit 200    # quick debug pass
.venv\Scripts\python.exe code\run_pipeline.py --zip          # build submission ZIP
```

Configuration comes from `.env` (`HYDE_API_KEY`, `HYDE_BASE_URL`, `HYDE_MODEL`).
The pipeline reads the key only at runtime — **no secrets are stored in code,
and `.env` is never included in the submission ZIP.**

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | Env-driven configuration, output paths, truncation limits |
| `schemas.py` | Pydantic v2 models for every emitted artifact |
| `extractor.py` | Parses nested message `parts`, OpenAI-style messages, `tool_calls` / `tool_calls_json`, `tool_call_response` results |
| `loader.py` | Streaming trace assembly: `iter_batches` + flush-on-completeness so memory is bounded by *incomplete* traces only |
| `batching.py` | Size-bounded classification batches (max 20 traces / 24k chars) |
| `llm_client.py` | Hyde GLM-5.3-Flash client; retries; verbatim response capture |
| `analytics.py` | Global aggregates with strict per-basis token separation |
| `validate.py` | Post-write re-validation of every artifact |
| `zip_build.py` | Submission ZIP with dataset/secret exclusion + verification |

## What it does

1. **Groups calls by `trace_id`, orders by `call_index`** — traces are flushed as
   soon as all `session_call_count` calls arrive; call order is verified
   (`call_order_preserved`).
2. **Extracts user messages, assistant responses, tool calls, and tool results**
   from all observed structures: message-level `parts` arrays
   (`text`/`toolCall`/`tool_call_response`/`thinking`), OpenAI-style
   `content` (string or typed list), `tool_calls`, and legacy `tool_calls_json`.
   Long texts are truncated with `truncated: true` + `original_length`.
3. **Preserves trace IDs and call metadata** — every record keeps
   `trace_id`, `span_id`, `parent_span_id`, `call_index`, `session_call_count`,
   `model`, `start_time`, `end_time`.
4. **Prepares classification batches** (`classification_batches.jsonl`) that
   reference the configured model; each contains a per-trace digest
   (first user message, final assistant message, tool names, token totals).
5. **No fabrication** — classifications exist *only* if `--classify` runs and
   stores verbatim API responses. Parse/validation failures are recorded as
   `validation_errors`, request failures as `error`; nothing is invented.
6. **Token usage kept separate by basis** — all aggregates are keyed by
   `usage_basis` (`source_reported` / `estimated_cl100k_base` / `unknown`);
   cache tokens and `source_reported_session_usage` are reported separately
   and never summed into per-call totals. Missing counts mean *unknown*, not 0.
7. **Validated outputs** — every artifact is re-read from disk and validated
   against the schemas; results land in `validation_report.json`.

## Outputs (`output/`)

| File | Content |
|---|---|
| `trace_calls.jsonl` | 50,000 normalized call records (metadata + extraction + usage) |
| `trace_summary.json` | 3,159 per-trace aggregates incl. `tokens_by_basis` |
| `classification_batches.jsonl` | 170 prepared batches (3,159 trace digests) |
| `classifications.jsonl` | Only real Hyde API outcomes — currently documents the live 401 attempt (invalid key); re-run `--classify` with a valid key to populate |
| `analytics.json` | Dataset-level stats, token totals per basis, tool frequency |
| `validation_report.json` | Schema-validation status of all files |

## Token accounting (final run)

| Basis | Calls | Input tokens | Output tokens |
|---|---|---|---|
| `source_reported` | 39,957 | 1,712,997,021 | 12,592,521 |
| `estimated_cl100k_base` | 9,000 | 148,083,201 | 1,739,830 |
| `unknown` | 1,043 | unknown | unknown |

1,043 calls have `usage_basis: unknown` — the source did not report counts;
they are excluded from token sums (never treated as zero). Cross-check:
39,957 + 9,000 = 48,957 known-token calls; sums match the dataset totals.

## Privacy & integrity notes

- Source data is historical; message content is never treated as instructions.
- The submission ZIP excludes `dataset/`, `.env`, `.venv/`, caches, and any
  file matching secret patterns; `zip_build.py` re-verifies the archive.
- See `dataset/SOURCES_AND_NOTICES.txt` for upstream sources and licenses;
  this build is for organizer review only.
