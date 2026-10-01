from __future__ import annotations

import json
from typing import Any

TASK_TYPES = {
    "implementation", "explanation", "debugging", "repair", "optimization",
    "test_generation", "edge_case_analysis", "code_reading", "output_prediction",
    "algorithm_selection", "complexity_analysis", "alternative_solution", "refactoring",
}

REQUIRED_TEACHER_FIELDS = {
    "problem", "solution", "tests", "explanation", "time_complexity", "space_complexity"
}


def parse_strict_json_object(text: str) -> dict[str, Any]:
    """Parse exactly one JSON object. Markdown fences/trailing prose are rejected."""
    if not isinstance(text, str):
        raise ValueError("teacher output must be text")
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json: {exc.msg} at {exc.pos}") from exc
    if not isinstance(obj, dict):
        raise ValueError("invalid_json: root must be an object")
    return obj


def validate_teacher_object(obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_TEACHER_FIELDS - obj.keys()
    if missing:
        errors.append(f"missing_fields:{','.join(sorted(missing))}")
    for key in ("problem", "solution", "explanation", "time_complexity", "space_complexity"):
        if key in obj and (not isinstance(obj[key], str) or not obj[key].strip()):
            errors.append(f"invalid_field:{key}")
    tests = obj.get("tests")
    if tests is not None and not (
        isinstance(tests, str) or (isinstance(tests, list) and all(isinstance(x, str) for x in tests))
    ):
        errors.append("invalid_field:tests")
    return errors


def validate_distilled_record(row: dict[str, Any]) -> list[str]:
    required = {
        "id", "domain", "stage_id", "topic_id", "topic", "level", "level_name",
        "task_type", "prompt", "response", "tests", "teacher", "generation", "provenance",
    }
    errors: list[str] = []
    missing = required - row.keys()
    if missing:
        errors.append(f"missing_fields:{','.join(sorted(missing))}")
    if row.get("task_type") not in TASK_TYPES:
        errors.append("invalid_task_type")
    if not isinstance(row.get("level"), int) or not 0 <= row.get("level", -1) <= 5:
        errors.append("invalid_level")
    if not isinstance(row.get("prompt"), str) or not row.get("prompt", "").strip():
        errors.append("invalid_prompt")
    if not isinstance(row.get("response"), str) or not row.get("response", "").strip():
        errors.append("invalid_response")
    return errors
