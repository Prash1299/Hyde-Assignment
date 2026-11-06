"""Pydantic schemas for every artifact the pipeline emits."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CallUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basis: str = Field(description="source_reported | estimated_cl100k_base | unknown")
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    token_estimation_note: Optional[str] = None

    @field_validator("basis")
    @classmethod
    def _basis_known(cls, v: str) -> str:
        allowed = {"source_reported", "estimated_cl100k_base", "unknown"}
        if v not in allowed:
            raise ValueError(f"unexpected usage_basis: {v!r}")
        return v


class ExtractedText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    text: str
    truncated: bool = False
    original_length: int = 0
    from_output: bool = False


class ExtractedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    arguments_is_json: bool = False
    partial: bool = False
    source: str


class ExtractedToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    result: Optional[str] = None
    truncated: bool = False
    original_length: int = 0


class CallExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_messages: List[ExtractedText] = Field(default_factory=list)
    assistant_messages: List[ExtractedText] = Field(default_factory=list)
    tool_calls: List[ExtractedToolCall] = Field(default_factory=list)
    tool_results: List[ExtractedToolResult] = Field(default_factory=list)
    system_message_count: int = 0
    thinking_part_count: int = 0
    malformed_message_count: int = 0
    message_counts: Dict[str, int] = Field(default_factory=dict)
    system_instructions_preview: Optional[str] = None


class CallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    call_index: int
    session_call_count: Optional[int] = None
    model: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    usage: CallUsage
    extraction: CallExtraction

    @field_validator("trace_id")
    @classmethod
    def _trace_id_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("trace_id must be non-empty")
        return v


class BasisTokenTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    known_input_calls: int = 0
    known_output_calls: int = 0


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    calls: int
    expected_calls: Optional[int] = None
    complete: bool
    models: Dict[str, int] = Field(default_factory=dict)
    usage_bases: Dict[str, int] = Field(default_factory=dict)
    tokens_by_basis: Dict[str, BasisTokenTotals] = Field(default_factory=dict)
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    calls_with_cache_fields: int = 0
    session_usage_source_reported: Optional[Dict[str, Any]] = None
    tool_call_count: int = 0
    distinct_tools: List[str] = Field(default_factory=list)
    first_user_message_preview: Optional[str] = None
    final_assistant_message_preview: Optional[str] = None
    first_call_index: Optional[int] = None
    last_call_index: Optional[int] = None
    earliest_start: Optional[str] = None
    latest_end: Optional[str] = None
    call_order_preserved: bool = True


class TraceDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    calls: int
    models: List[str] = Field(default_factory=list)
    tool_names: List[str] = Field(default_factory=list)
    first_user_message: Optional[str] = None
    final_assistant_message: Optional[str] = None
    tokens_by_basis: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class ClassificationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    model: str
    created_at: str
    status: str
    trace_count: int
    traces: List[TraceDigest]

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in {"prepared", "classified"}:
            raise ValueError(f"unexpected batch status: {v!r}")
        return v


class ClassificationResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    primary_intent: str
    domain: str
    uses_tools: bool
    outcome: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

    @field_validator("primary_intent", "domain", "outcome", "rationale")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("classification fields must be non-empty")
        return v


class ClassificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    trace_id: str
    model: str
    created_at: str
    response_raw: str
    parsed: Optional[Dict[str, Any]] = None
    validated: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    api_usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Analytics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    total_calls: int = 0
    total_traces: int = 0
    complete_traces: int = 0
    incomplete_traces: int = 0
    row_groups: int = 0
    models: Dict[str, int] = Field(default_factory=dict)
    usage_bases: Dict[str, int] = Field(default_factory=dict)
    tokens_by_basis: Dict[str, BasisTokenTotals] = Field(default_factory=dict)
    cache_read_input_tokens_total: int = 0
    cache_creation_input_tokens_total: int = 0
    calls_with_cache_fields: int = 0
    session_usage_records: int = 0
    tool_call_count: int = 0
    distinct_tools: List[str] = Field(default_factory=list)
    extraction_coverage: Dict[str, int] = Field(default_factory=dict)
    classification: Dict[str, Any] = Field(default_factory=dict)
    generated_at: str = ""
