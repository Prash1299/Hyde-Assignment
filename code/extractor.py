"""Extraction of user messages, assistant responses, and tool calls from
the heterogeneous nested message structures found in the dataset.

Supported shapes (observed in traces.parquet):
  * message-level ``parts`` arrays with typed entries:
      {type: text, content|text}, {type: thinking, thinking},
      {type: toolCall|tool_call, id, name, arguments|partialJson},
      {type: tool_call_response, id, result}
  * OpenAI-style messages: content str | list[{type: text, text}],
    tool_calls: [{id, type: function, function: {name, arguments(JSON str)}}]
    plus a legacy ``tool_calls_json`` JSON-string field.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from config import Limits
from schemas import CallExtraction, ExtractedText, ExtractedToolCall, ExtractedToolResult


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _clip(text: Optional[str], limit: int) -> Tuple[Optional[str], bool, int]:
    if text is None:
        return None, False, 0
    original = len(text)
    if original <= limit:
        return text, False, original
    return text[:limit], True, original


def _part_text(part: Any) -> Optional[str]:
    if not isinstance(part, dict):
        return _as_str(part)
    for key in ("content", "text"):
        if part.get(key) is not None:
            return _as_str(part[key])
    return None


def _add_text(target: List[ExtractedText], text: Optional[str], index: int, from_output: bool, limit: int) -> None:
    clipped, truncated, original = _clip(text, limit)
    if clipped:
        target.append(
            ExtractedText(index=index, text=clipped, truncated=truncated, original_length=original, from_output=from_output)
        )


def _parse_tool_calls_field(raw: Any, source: str, out: List[ExtractedToolCall]) -> None:
    if raw is None:
        return
    items = raw
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except (ValueError, TypeError):
            return
    if not isinstance(items, list):
        return
    for tc in items:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = func.get("name") or tc.get("name")
        args = func.get("arguments", tc.get("arguments"))
        partial_json = tc.get("partialJson")
        args_str: Optional[str] = None
        args_is_json = False
        if isinstance(args, str):
            args_str = args
            args_is_json = True
        elif args is not None:
            args_str = _as_str(args)
            args_is_json = isinstance(args, (dict, list))
        elif partial_json is not None:
            args_str = _as_str(partial_json)
        out.append(
            ExtractedToolCall(
                id=tc.get("id"),
                name=name,
                arguments=args_str,
                arguments_is_json=args_is_json,
                partial=partial_json is not None,
                source=source,
            )
        )


def _walk_message(
    msg: Any,
    from_output: bool,
    limits: Limits,
    extraction: CallExtraction,
    idx_counters: Dict[str, int],
) -> None:
    if not isinstance(msg, dict):
        extraction.malformed_message_count += 1
        return

    role = msg.get("role")

    if role == "system":
        extraction.system_message_count += 1

    parts = msg.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                if role == "user":
                    _add_text(extraction.user_messages, _part_text(part), idx_counters["user"], from_output, limits.text_preview_chars)
                    idx_counters["user"] += 1
                elif role == "assistant":
                    _add_text(extraction.assistant_messages, _part_text(part), idx_counters["assistant"], from_output, limits.text_preview_chars)
                    idx_counters["assistant"] += 1
            elif ptype in ("toolCall", "tool_call"):
                args = part.get("arguments")
                if args is None:
                    args = part.get("partialJson")
                extraction.tool_calls.append(
                    ExtractedToolCall(
                        id=part.get("id"),
                        name=part.get("name"),
                        arguments=_as_str(args),
                        arguments_is_json=isinstance(args, (dict, list)),
                        partial=part.get("arguments") is None and part.get("partialJson") is not None,
                        source="parts",
                    )
                )
            elif ptype == "tool_call_response":
                clipped, truncated, original = _clip(_as_str(part.get("result")), limits.text_preview_chars)
                extraction.tool_results.append(
                    ExtractedToolResult(
                        tool_call_id=part.get("id"),
                        tool_name=msg.get("tool_name"),
                        result=clipped,
                        truncated=truncated,
                        original_length=original,
                    )
                )
            elif ptype == "thinking":
                extraction.thinking_part_count += 1

    content = msg.get("content")
    if role not in ("tool", "system"):
        if isinstance(content, str):
            _add_text(extraction.user_messages if role == "user" else extraction.assistant_messages,
                      content, idx_counters[role or "unknown"], from_output, limits.text_preview_chars)
            idx_counters[role or "unknown"] += 1
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    _add_text(extraction.user_messages if role == "user" else extraction.assistant_messages,
                              _part_text(part), idx_counters[role or "unknown"], from_output, limits.text_preview_chars)
                    idx_counters[role or "unknown"] += 1
    elif role == "tool" and isinstance(content, str):
        clipped, truncated, original = _clip(content, limits.text_preview_chars)
        extraction.tool_results.append(
            ExtractedToolResult(
                tool_call_id=msg.get("tool_call_id"),
                tool_name=msg.get("tool_name"),
                result=clipped,
                truncated=truncated,
                original_length=original,
            )
        )

    _parse_tool_calls_field(msg.get("tool_calls"), "tool_calls", extraction.tool_calls)
    _parse_tool_calls_field(msg.get("tool_calls_json"), "tool_calls_json", extraction.tool_calls)


def extract_call(row: Dict[str, Any], limits: Limits) -> CallExtraction:
    extraction = CallExtraction()
    idx_counters = {"user": 0, "assistant": 0, "unknown": 0}

    for msg in row.get("input_messages") or []:
        _walk_message(msg, False, limits, extraction, idx_counters)
    for msg in row.get("output_messages") or []:
        _walk_message(msg, True, limits, extraction, idx_counters)

    extraction.message_counts = {
        "input": len(row.get("input_messages") or []),
        "output": len(row.get("output_messages") or []),
    }

    seen = set()
    deduped = []
    for tc in extraction.tool_calls:
        key = (tc.id, tc.name, tc.arguments)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tc)
    extraction.tool_calls = deduped

    for key in ("user_messages", "assistant_messages", "tool_calls", "tool_results"):
        items = getattr(extraction, key)
        cap = limits.max_tool_items if key in ("tool_calls", "tool_results") else limits.max_messages_per_list
        if len(items) > cap:
            del items[cap:]

    extra = row.get("extra")
    if isinstance(extra, dict) and extra.get("system_instructions") is not None:
        preview, _, _ = _clip(_as_str(extra.get("system_instructions")), limits.system_instructions_chars)
        extraction.system_instructions_preview = preview

    return extraction
