from __future__ import annotations
import json


def judge_prompt(row: dict) -> str:
    slim = {k: row.get(k) for k in ("domain", "topic_id", "level", "task_type", "prompt", "answer", "response", "tests")}
    schema = {
        "correctness": 0.0,
        "clarity": 0.0,
        "test_quality": 0.0,
        "curriculum_fit": 0.0,
        "completeness": 0.0,
        "conceptual_consistency": 0.0,
        "notes": "",
    }
    return (
        "You are an independent coding-dataset judge. Score the sample without trusting its teacher.\n"
        "Return ONLY one JSON object with numeric values 0..1 using this schema:\n"
        + json.dumps(schema)
        + "\nSample:\n"
        + json.dumps(slim, ensure_ascii=False)
    )


def test_generation_prompt(row: dict) -> str:
    return (
        "Generate independent adversarial Python tests for this problem. Do not inspect or imitate teacher tests.\n"
        'Return ONLY JSON: {"tests":["assert ...", "..."]}\nProblem:\n'
        + row.get("prompt", "")
    )
