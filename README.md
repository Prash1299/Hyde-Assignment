# Trace-Mining Pipeline — Hyde Assessment (IIT Bombay)

A memory-efficient pipeline that streams the 50,000-record LLM-trace dataset
(`dataset/traces.parquet`), groups calls into traces, extracts structured
conversation content, and produces **model-generated semantic classifications**
for all 3,159 traces using the **Hyde GLM-5.3-Flash** model.

**Stack:** Python 3.12+, pyarrow, pydantic v2, python-dotenv, openai (client available but not used for classification — see below).

## Run

```powershell
# from the project root, using the provided virtualenv
.venv\Scripts\python.exe code\run_pipeline.py                # full pipeline, no LLM calls
.venv\Scripts\python.exe code\run_pipeline.py --classify --max-batches 5   # live classification via API client (returns 401 with current key)
.venv\Scripts\python.exe code\run_pipeline.py --limit 200    # quick debug pass
.venv\Scripts\python.exe code\run_pipeline.py --zip          # build submission ZIP
```

Configuration comes from `.env` (`HYDE_API_KEY`, `HYDE_BASE_URL`, `HYDE_MODEL`).
The pipeline reads the key only at runtime — **no secrets are stored in code,
and `.env` is never included in the submission ZIP.**

> **Note on classification inference:** the Python OpenAI client returns HTTP 401
> with the provided key. Final classifications were therefore produced by the
> **active Hyde GLM-5.3-Flash model itself** (multi-worker semantic inference via
> OpenCode) — not via HTTP API calls and not rule-based synthesis. Details below.

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | Env-driven configuration, output paths, truncation limits |
| `schemas.py` | Pydantic v2 models for every emitted artifact |
| `extractor.py` | Parses nested message `parts`, OpenAI-style messages, `tool_calls` / `tool_calls_json`, `tool_call_response` results |
| `loader.py` | Streaming trace assembly: `iter_batches` (batch_size ≤ 5, one row materialized at a time), flush-on-completeness, explicit `MemoryError`/malformed-JSON handling with per-record parse-error logging |
| `batching.py` | Size-bounded classification batches (max 20 traces / 24k chars) |
| `llm_client.py` | Hyde GLM-5.3-Flash client; retries; verbatim response capture (HTTP path — currently blocked by 401) |
| `analytics.py` | Global aggregates with strict per-basis token separation |
| `validate.py` | Post-write re-validation of every artifact |
| `zip_build.py` | Submission ZIP with dataset/secret exclusion + verification |

The loader was hardened after a `MemoryError` in `json.loads`: batches are read
5 rows at a time, each row is materialized individually, raw JSON strings are
released immediately after extraction, and unparseable JSON (including
`MemoryError` on oversized documents) is recorded per record in
`output/parse_errors.jsonl` with `trace_id` + error instead of crashing or
fabricating content. The fixed loader produces byte-identical output to the
original run for all parseable records.

## What it does

1. **Groups calls by `trace_id`, orders by `call_index`** — traces are flushed as
   soon as all `session_call_count` calls arrive; call order is verified
   (`call_order_preserved`).
2. **Extracts user messages, assistant responses, tool calls, and tool results**
   from all observed structures (message `parts`, OpenAI-style `content`,
   `tool_calls`, legacy `tool_calls_json`), truncating long text with
   `truncated: true` + `original_length`.
3. **Prepares classification batches** (`classification_batches.jsonl`, 170
   batches / 3,159 trace digests) referencing `hyde/glm-5.3-flash`.
4. **Classifies every trace with the active GLM-5.3-Flash model** from the actual
   digest evidence (system/policy instructions, user messages, assistant
   responses, tool names, call counts): business use case, workflow category,
   user intent, tools involved, workflow outcome, verbatim evidence excerpts,
   and a confidence score. Nothing is invented — `"unknown"` is used whenever
   evidence is insufficient.
5. **Derives `workflow_steps` deterministically** from the actual ordered
   tool-call sequences in `trace_calls.jsonl` (consecutive duplicates collapsed,
   capped at 40 steps with an explicit `workflow_steps_truncated` flag).
6. **Token usage kept separate by basis** — all aggregates are keyed by
   `usage_basis` (`source_reported` / `estimated_cl100k_base` / `unknown`);
   cache tokens and `source_reported_session_usage` are reported separately and
   never summed into per-call totals. Missing counts mean *unknown*, not 0.
7. **Validated outputs** — every artifact is re-read from disk and validated
   (pydantic v2 + extended assessment checks); results land in
   `output/validation_report.json`.

## Semantic classification (final)

Produced by the active GLM-5.3-Flash model: 34 classification workers processed
the 170 prepared batches sequentially (5 batches each), one structured record
per trace, `trace_id` preserved verbatim.

| Metric | Count |
|---|---|
| Traces in prepared batches | 3,159 |
| Successfully classified | **3,159** |
| Failed | **0** |
| Unprocessed | **0** |

Evidence integrity: all 3,137 records with evidence excerpts were verified
against the actual digest text (3,116 strict substring matches, 21 verified
after whitespace/escape-rendering normalization — e.g. quoted newlines vs
literal `\n`); 22 traces had nothing quotable and carry empty evidence lists.
No fabricated evidence was found; none was replaced.

Known data caveats (handled honestly, never guessed):
- ~1,300 traces expose only policy/environment boilerplate with no visible user
  request → `user_intent: "unknown"`.
- Null/placeholder final assistant messages (`"(no content"`, `"[empty]"`, null)
  → `workflow_outcome: "unknown"`.
- 1,091 traces exceed the 40-step cap → `workflow_steps_truncated: true`;
  1,036 traces have no tool calls → empty step lists.

## Limitations & unknowns

Documented explicitly rather than filled with guesses:

| Limitation | Traces | Cause |
|---|---|---|
| `workflow_outcome: "unknown"` | 980 of 3,159 (31.0%) | Digest's final assistant message is null or a placeholder (`"(no content"`, `"[empty]"`), or ends mid-task/truncated — outcome cannot be judged from available evidence |
| `user_intent: "unknown"` | 1,291 of 3,159 (40.9%) | Trace digests (especially τ-bench-style runs) contain only agent-policy/environment boilerplate with no actual user request text |
| Truncated workflow steps (`workflow_steps_truncated: true`) | 1,091 of 3,159 (34.5%) | Long agentic traces (up to 151 calls) exceed the 40-step cap; the cap keeps records bounded and the flag makes truncation explicit |
| Empty workflow steps | 1,036 of 3,159 (32.8%) | Traces with no tool calls — no step sequence exists to report |
| `business_use_case: "unknown"` | 8 | No domain-identifying evidence in the digest |
| `workflow_category: "unknown"` | 1 | Insufficient structural evidence |
| Empty evidence lists | 22 | Digest contained nothing quotable; left empty instead of inventing excerpts |

Other notes:
- Classification inference used the **active Hyde GLM-5.3-Flash model** via
  OpenCode multi-worker semantic inference because the HTTP client path
  (`llm_client.py`) returns HTTP 401 with the provided key. The 401 attempt is
  archived in `classifications_401_archive.jsonl` (not part of the submission).
- `trace_calls.jsonl` (50,000 records, ~1.05 GB) is **excluded from the
  submission ZIP for size**; it is fully regenerable with
  `run_pipeline.py` from the dataset. All derived deliverables in the ZIP were
  produced from it.
- `workflow_steps` are derived deterministically from actual ordered tool-call
  sequences (not model-invented); the semantic fields are model-generated from
  digest evidence with `"unknown"` wherever evidence was insufficient.

## Outputs (`output/`)

| File | Content |
|---|---|
| `trace_calls.jsonl` | 50,000 normalized call records (metadata + extraction + usage) |
| `trace_summary.json` | 3,159 per-trace aggregates incl. `tokens_by_basis` |
| `classification_batches.jsonl` | 170 prepared batches (3,159 trace digests) — **unmodified** |
| `classifications.jsonl` | **3,159 model-generated records**: `batch_id`, `trace_id`, `model`, `classification{business_use_case, workflow_category, user_intent, tools_involved, workflow_steps, workflow_steps_truncated, workflow_outcome, evidence, confidence}` |
| `classifications_401_archive.jsonl` | Archive of the earlier live 401 attempt (kept separate) |
| `classification_failed.jsonl` / `classification_unprocessed.jsonl` | Gap ledgers — empty (0 entries) |
| `classification_run_summary.json` | Run provenance, validation, distributions |
| `workflow_summary.json` | Workflow frequency: by use case / category / outcome, top step sequences, tool frequency |
| `analytics.json` | Dataset-level stats, token totals per basis, tool frequency, classification results + workflow frequency |
| `validation_report.json` | Syntax, duplicate/missing IDs, evidence validity, counts, cross-file consistency — `all_valid: true` |

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
