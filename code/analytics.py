"""Global analytics accumulation with strict separation of token-usage bases."""

from collections import Counter
from typing import Dict, Optional

from schemas import Analytics, BasisTokenTotals, TraceRecord


class AnalyticsAccumulator:
    def __init__(self, dataset_label: str):
        self.analytics = Analytics(dataset=dataset_label)
        self._tool_counter: Counter = Counter()

    def add_trace(self, trace: TraceRecord) -> None:
        a = self.analytics
        a.total_traces += 1
        if trace.complete:
            a.complete_traces += 1
        else:
            a.incomplete_traces += 1
        for model, count in trace.models.items():
            a.models[model] = a.models.get(model, 0) + count
        for basis, count in trace.usage_bases.items():
            a.usage_bases[basis] = a.usage_bases.get(basis, 0) + count
        for basis, totals in trace.tokens_by_basis.items():
            bucket = a.tokens_by_basis.setdefault(basis, BasisTokenTotals())
            bucket.calls += totals.calls
            bucket.input_tokens += totals.input_tokens
            bucket.output_tokens += totals.output_tokens
            bucket.known_input_calls += totals.known_input_calls
            bucket.known_output_calls += totals.known_output_calls
        if trace.cache_read_input_tokens is not None:
            a.cache_read_input_tokens_total += trace.cache_read_input_tokens
        if trace.cache_creation_input_tokens is not None:
            a.cache_creation_input_tokens_total += trace.cache_creation_input_tokens
        a.calls_with_cache_fields += trace.calls_with_cache_fields
        if trace.session_usage_source_reported is not None:
            a.session_usage_records += 1
        a.tool_call_count += trace.tool_call_count
        for tool in trace.distinct_tools:
            self._tool_counter[tool] += 1

    def add_call_flags(self, has_user: bool, has_assistant: bool, has_tool_calls: bool, has_tool_results: bool, malformed: int) -> None:
        cov = self.analytics.extraction_coverage
        if has_user:
            cov["calls_with_user_messages"] = cov.get("calls_with_user_messages", 0) + 1
        if has_assistant:
            cov["calls_with_assistant_messages"] = cov.get("calls_with_assistant_messages", 0) + 1
        if has_tool_calls:
            cov["calls_with_tool_calls"] = cov.get("calls_with_tool_calls", 0) + 1
        if has_tool_results:
            cov["calls_with_tool_results"] = cov.get("calls_with_tool_results", 0) + 1
        if malformed:
            cov["calls_with_malformed_messages"] = cov.get("calls_with_malformed_messages", 0) + 1

    def finalize(self, row_groups: int, total_calls: int, generated_at: str, classification_summary: Optional[Dict] = None) -> Analytics:
        a = self.analytics
        a.row_groups = row_groups
        a.total_calls = total_calls
        a.generated_at = generated_at
        a.distinct_tools = [name for name, _ in self._tool_counter.most_common()]
        if classification_summary is not None:
            a.classification = classification_summary
        else:
            a.classification = {"executed": False, "note": "classification stage not run; no classifications generated"}
        return a
