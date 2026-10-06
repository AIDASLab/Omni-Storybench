"""Deterministic parsing and normalization for judge-model responses."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_SMART_QUOTES = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
)


def _key_signature(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _close_truncated_json_containers(candidate: str) -> Optional[str]:
    stack: List[str] = []
    in_string = False
    escaped = False

    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener, char) not in {("{", "}"), ("[", "]")}:
                return None

    if in_string or not stack:
        return None

    repaired = candidate.rstrip()
    if not repaired:
        return None
    if repaired.endswith(","):
        repaired = repaired[:-1].rstrip()
    if not repaired or repaired.endswith(":"):
        return None

    closers = "".join(
        "}" if opener == "{" else "]" for opener in reversed(stack)
    )
    return repaired + closers


def _balanced_objects(text: str) -> Iterable[str]:
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def _candidate_json_strings(raw_response: str) -> List[str]:
    stripped = raw_response.strip()
    candidates: List[str] = []
    if stripped:
        candidates.append(stripped)

    for fenced in re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.I | re.S):
        if fenced.strip():
            candidates.append(fenced.strip())

    candidates.extend(_balanced_objects(stripped))

    deduplicated: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduplicated.append(candidate)
    return deduplicated


def _next_non_whitespace(text: str, start: int) -> int:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _looks_like_object_member(text: str, start: int) -> bool:
    index = _next_non_whitespace(text, start)
    if index >= len(text) or text[index] == "}":
        return True
    if text[index] != '"':
        return False

    index += 1
    escaped = False
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            following = _next_non_whitespace(text, index + 1)
            return following < len(text) and text[following] == ":"
        index += 1
    return False


def _looks_like_array_value(text: str, start: int) -> bool:
    index = _next_non_whitespace(text, start)
    if index >= len(text) or text[index] == "]":
        return True
    if text[index] in {'"', "{", "[", "-"} or text[index].isdigit():
        return True
    for literal in ("true", "false", "null"):
        if text.startswith(literal, index):
            end = _next_non_whitespace(text, index + len(literal))
            return end >= len(text) or text[end] in {",", "]", "}"}
    return False


def _quote_can_close_string(
    text: str,
    quote_index: int,
    role: str,
    container: Optional[str],
) -> bool:
    following = _next_non_whitespace(text, quote_index + 1)
    if role == "key":
        return following < len(text) and text[following] == ":"
    if following >= len(text):
        return True

    token = text[following]
    if container == "{":
        if token == "}":
            return True
        return token == "," and _looks_like_object_member(text, following + 1)
    if container == "[":
        if token == "]":
            return True
        return token == "," and _looks_like_array_value(text, following + 1)
    return False


def _escape_unescaped_quotes_in_json_strings(candidate: str) -> str:
    """Escape quotes that cannot legally terminate their current JSON string."""

    output: List[str] = []
    containers: List[str] = []
    in_string = False
    escaped = False
    string_role = "value"
    string_container: Optional[str] = None
    previous_token: Optional[str] = None

    for index, char in enumerate(candidate):
        if in_string:
            if escaped:
                output.append(char)
                escaped = False
            elif char == "\\":
                output.append(char)
                escaped = True
            elif char == '"':
                if _quote_can_close_string(
                    candidate,
                    index,
                    string_role,
                    string_container,
                ):
                    output.append(char)
                    in_string = False
                    previous_token = '"'
                else:
                    output.append('\\"')
            else:
                output.append(char)
            continue

        output.append(char)
        if char == '"':
            string_container = containers[-1] if containers else None
            string_role = (
                "key"
                if string_container == "{" and previous_token in {"{", ","}
                else "value"
            )
            in_string = True
            escaped = False
        elif char in "{[":
            containers.append(char)
            previous_token = char
        elif char in "}]":
            expected = "{" if char == "}" else "["
            if containers and containers[-1] == expected:
                containers.pop()
            previous_token = char
        elif not char.isspace():
            previous_token = char

    return "".join(output)


def _normalized_json_strings(candidate: str) -> Iterable[str]:
    translated = candidate.translate(_SMART_QUOTES).strip()
    translated = re.sub(r",\s*([}\]])", r"\1", translated)

    variants = [translated]
    quote_repaired = _escape_unescaped_quotes_in_json_strings(translated)
    if quote_repaired != translated:
        variants.append(quote_repaired)

    # Some models emit Python literals even when JSON was explicitly requested.
    for variant in list(variants):
        literals = re.sub(r"\bTrue\b", "true", variant)
        literals = re.sub(r"\bFalse\b", "false", literals)
        literals = re.sub(r"\bNone\b", "null", literals)
        if literals not in variants:
            variants.append(literals)

    for normalized in variants:
        yield normalized
        repaired = _close_truncated_json_containers(normalized)
        if repaired is not None:
            yield repaired


def _find_required_value(value: Any, signature: str) -> Optional[Any]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _key_signature(str(key)) == signature:
                return nested
        for nested in value.values():
            found = _find_required_value(nested, signature)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_required_value(nested, signature)
            if found is not None:
                return found
    return None


def _normalize_score(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*(?:10|[1-9])(?:\.0+)?\s*", value):
        return int(float(value))
    return None


def _normalize_rationale(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_category_value(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None

    score = None
    rationale = None
    for key, nested in value.items():
        signature = _key_signature(str(key))
        if signature in {"score", "rating", "value"}:
            score = _normalize_score(nested)
        elif signature in {"rationale", "reason", "explanation", "justification"}:
            rationale = _normalize_rationale(nested)

    if score is None or not 1 <= score <= 10 or rationale is None:
        return None
    return {"score": score, "rationale": rationale}


def _validate_exact_schema(
    value: Any,
    categories: Sequence[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(value, Mapping):
        return None, "Top-level response is not a JSON object"

    if set(value.keys()) != set(categories):
        missing = [category for category in categories if category not in value]
        extra = [str(key) for key in value if key not in categories]
        return None, f"Schema key mismatch; missing={missing}, extra={extra}"

    result: Dict[str, Any] = {}
    for category in categories:
        item = value[category]
        if not isinstance(item, Mapping):
            return None, f"{category!r} is not an object"
        if set(item.keys()) != {"score", "rationale"}:
            return None, f"{category!r} must contain only score and rationale"
        score = item["score"]
        rationale = item["rationale"]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 10:
            return None, f"{category!r}.score must be an integer from 1 to 10"
        if not isinstance(rationale, str) or not rationale.strip():
            return None, f"{category!r}.rationale must be a non-empty string"
        result[category] = {"score": score, "rationale": rationale.strip()}
    return result, None


def _normalize_schema(
    value: Any,
    categories: Sequence[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(value, (Mapping, list)):
        return None, "Parsed value cannot contain the requested schema"

    normalized: Dict[str, Any] = {}
    failures: List[str] = []
    for category in categories:
        signature = _key_signature(category)
        nested = _find_required_value(value, signature)
        if nested is None and signature == "satisfactionofgenerationconditions":
            nested = _find_required_value(value, "atisfactionofgenerationconditions")
        item = _normalize_category_value(nested)
        if item is None:
            failures.append(category)
        else:
            normalized[category] = item

    if failures:
        return None, f"Missing or invalid category payloads: {failures}"
    return normalized, None


def parse_judge_response(
    raw_response: Optional[str],
    categories: Sequence[str],
) -> Dict[str, Any]:
    raw = raw_response if isinstance(raw_response, str) else ""
    base: Dict[str, Any] = {
        "inference_status": "ok",
        "raw_response": raw_response,
        "judge_eval": None,
        "parse_method": None,
        "repair_status": "pending",
        "repair_raw_response": None,
        "initial_parse_error": None,
        "final_parse_error": None,
        "parse_error": None,
    }

    if not raw.strip():
        error = "Model returned an empty response"
        base.update(
            {
                "status": "parse_error",
                "parse_status": "unresolved",
                "initial_parse_error": error,
                "final_parse_error": error,
                "parse_error": error,
            }
        )
        return base

    errors: List[str] = []
    try:
        direct_value = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError) as exc:
        direct_value = None
        errors.append(f"direct JSON decode failed: {exc}")
    else:
        validated, error = _validate_exact_schema(direct_value, categories)
        if validated is not None:
            base.update(
                {
                    "status": "ok",
                    "parse_status": "direct",
                    "parse_method": "direct",
                    "repair_status": "not_needed",
                    "judge_eval": validated,
                }
            )
            return base
        errors.append(error or "direct schema validation failed")

    parsed_values: List[Any] = []
    if direct_value is not None:
        parsed_values.append(direct_value)

    for candidate in _candidate_json_strings(raw):
        for normalized_candidate in _normalized_json_strings(candidate):
            try:
                parsed_value = json.loads(normalized_candidate)
            except (json.JSONDecodeError, TypeError):
                try:
                    parsed_value = ast.literal_eval(normalized_candidate)
                except (SyntaxError, ValueError):
                    continue
            parsed_values.append(parsed_value)

    seen_values = set()
    for value in parsed_values:
        try:
            fingerprint = json.dumps(value, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            fingerprint = repr(value)
        if fingerprint in seen_values:
            continue
        seen_values.add(fingerprint)

        normalized, error = _normalize_schema(value, categories)
        if normalized is not None:
            initial_error = "; ".join(errors) if errors else None
            base.update(
                {
                    "status": "ok",
                    "parse_status": "normalized",
                    "parse_method": "normalized",
                    "repair_status": "not_needed",
                    "judge_eval": normalized,
                    "initial_parse_error": initial_error,
                }
            )
            return base
        if error:
            errors.append(error)

    error = "; ".join(dict.fromkeys(errors)) or "No valid JSON object found"
    base.update(
        {
            "status": "parse_error",
            "parse_status": "unresolved",
            "initial_parse_error": error,
            "final_parse_error": error,
            "parse_error": error,
        }
    )
    return base


def build_repair_messages(
    raw_response: str,
    parse_error: str,
    categories: Sequence[str],
) -> List[Dict[str, str]]:
    schema = _schema_for_repair(categories)
    return [
        {
            "role": "system",
            "content": (
                "You repair malformed judge output without inventing scores or "
                "rationales. Preserve the source meaning exactly."
            ),
        },
        {
            "role": "user",
            "content": f"""Convert the response below into exactly the requested JSON schema.

Rules:
- Never add a score or rationale that is absent from the source.
- Correct only formatting, key spelling, nesting, and obvious JSON/type errors.
- If every required value cannot be recovered, return exactly {{"unrecoverable": true}}.
- Return JSON only, with no Markdown or surrounding prose.

Required schema:
{schema}

Initial parse error:
{parse_error}

Raw response:
{raw_response}
""",
        },
    ]


def _schema_for_repair(categories: Sequence[str]) -> str:
    body = ",\n".join(
        f'  "{category}": {{"score": <integer 1-10>, "rationale": "<non-empty string>"}}'
        for category in categories
    )
    return "{\n" + body + "\n}"


def build_judge_json_schema(categories: Sequence[str]) -> Dict[str, Any]:
    category_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 10},
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": ["score", "rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {category: category_schema for category in categories},
        "required": list(categories),
        "additionalProperties": False,
    }


def build_repair_json_schema(categories: Sequence[str]) -> Dict[str, Any]:
    unrecoverable_schema = {
        "type": "object",
        "properties": {"unrecoverable": {"const": True}},
        "required": ["unrecoverable"],
        "additionalProperties": False,
    }
    return {
        "oneOf": [
            build_judge_json_schema(categories),
            unrecoverable_schema,
        ]
    }


def is_unrecoverable_response(raw_response: Optional[str]) -> bool:
    if not isinstance(raw_response, str):
        return False
    for candidate in _candidate_json_strings(raw_response):
        for normalized_candidate in _normalized_json_strings(candidate):
            try:
                value = json.loads(normalized_candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, Mapping) and value.get("unrecoverable") is True:
                return True
    return False


def runtime_error_fields(error: str, raw_response: Optional[str] = None) -> Dict[str, Any]:
    return {
        "status": "error",
        "inference_status": "error",
        "parse_status": "not_attempted",
        "repair_status": "not_eligible",
        "parse_method": None,
        "judge_eval": None,
        "raw_response": raw_response,
        "initial_parse_error": None,
        "repair_raw_response": None,
        "final_parse_error": None,
        "parse_error": None,
        "error": error,
    }
