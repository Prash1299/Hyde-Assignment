"""Pipeline configuration, loaded from environment (.env supported)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "traces.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRACE_CALLS_PATH = OUTPUT_DIR / "trace_calls.jsonl"
TRACE_SUMMARY_PATH = OUTPUT_DIR / "trace_summary.json"
BATCHES_PATH = OUTPUT_DIR / "classification_batches.jsonl"
CLASSIFICATIONS_PATH = OUTPUT_DIR / "classifications.jsonl"
ANALYTICS_PATH = OUTPUT_DIR / "analytics.json"
VALIDATION_REPORT_PATH = OUTPUT_DIR / "validation_report.json"
SUBMISSION_ZIP_PATH = PROJECT_ROOT / "hyde_trace_mining_submission.zip"


@dataclass
class Limits:
    text_preview_chars: int = 1200
    digest_first_user_chars: int = 800
    digest_final_assistant_chars: int = 600
    system_instructions_chars: int = 300
    max_messages_per_list: int = 20
    max_tool_items: int = 40
    max_batch_traces: int = 20
    max_batch_chars: int = 24000


@dataclass
class HydeConfig:
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 900
    request_timeout_s: float = 120.0
    max_retries: int = 3
    limits: Limits = field(default_factory=Limits)


LIMITS_DEFAULTS = Limits()


def load_config(require_api_key: bool = False) -> HydeConfig:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    cfg = HydeConfig(
        api_key=os.getenv("HYDE_API_KEY"),
        base_url=os.getenv("HYDE_BASE_URL"),
        model=os.getenv("HYDE_MODEL"),
    )
    if require_api_key:
        missing = [k for k, v in (("HYDE_API_KEY", cfg.api_key), ("HYDE_BASE_URL", cfg.base_url), ("HYDE_MODEL", cfg.model)) if not v]
        if missing:
            raise ValueError(f"Missing Hyde GLM configuration in .env: {', '.join(missing)}")
    return cfg
