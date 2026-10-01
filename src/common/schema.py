from __future__ import annotations
from dataclasses import dataclass
from typing import Any

REQUIRED = {"id", "source", "language", "category", "skill", "difficulty", "task_type", "prompt"}
VALID_TASK_TYPES = {
    "implementation", "debugging", "optimization", "explanation", "test_generation",
    "code_review", "tool_use", "repair", "agent"
}


def validate_record(x: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED - x.keys()
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if x.get("language") not in {"python", "text", None}:
        errors.append("language must be python or text")
    if "difficulty" in x and not isinstance(x["difficulty"], int):
        errors.append("difficulty must be int")
    if x.get("task_type") and x["task_type"] not in VALID_TASK_TYPES:
        errors.append(f"unknown task_type={x['task_type']}")
    if not isinstance(x.get("prompt", ""), str) or not x.get("prompt", "").strip():
        errors.append("prompt must be non-empty")
    return errors
