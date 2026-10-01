from __future__ import annotations

from src.distill.curriculum import CurriculumCell

SYSTEM = """You are a dataset-generation teacher creating rigorous supervised data for a Python specialist model.
Return EXACTLY one valid JSON object. Do not use markdown fences and do not write prose outside JSON.
The JSON object MUST have exactly these semantic fields:
problem, solution, tests, explanation, time_complexity, space_complexity.

Rules:
- The content must match the requested curriculum topic, level, and task type.
- Use Python unless the task is purely explanatory.
- Make the example novel, precise, self-contained, and unambiguous.
- Tests must be deterministic and offline. No network, external services, sleeps, random dependence, or destructive operations.
- For executable tasks, the solution should be executable Python and tests should meaningfully check edge cases.
- For debugging/repair tasks, the problem must include the broken code or failure scenario and the solution must contain the corrected code.
- Do not claim test results that were not executed; simply provide test cases.
"""


def build_generation_prompt(cell: CurriculumCell, sample_index: int, nonce: str) -> str:
    return f"""Generate one training example.

CURRICULUM
Domain: {cell.domain}
Stage: {cell.stage_id} ({cell.stage_name})
Topic: {cell.topic_id} ({cell.topic_name})
Level: L{cell.level} ({cell.level_name})
Task type: {cell.task_type}
Sample index: {sample_index}
Diversity nonce: {nonce}

Difficulty guidance:
L0 = teach/recognize the concept with minimal prerequisites.
L1 = straightforward beginner application.
L2 = combine the concept with realistic constraints.
L3 = advanced application with traps or multiple steps.
L4 = hard, efficiency-sensitive, adversarial edge cases.
L5 = expert/competitive/Olympiad-style where appropriate.

Return only the JSON object."""
