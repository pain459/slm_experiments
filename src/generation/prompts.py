SYSTEM = """You are generating high-quality supervised data for a Python coding model.
Return exactly one JSON object and no markdown. The object must contain:
problem, solution, tests, category, skill, difficulty, task_type, complexity.
The solution must be self-contained Python when the task is executable.
Tests must be Python assertion strings or a single pytest-compatible test module.
Avoid network access, external services, nondeterminism, and destructive operations."""

REPAIR_SYSTEM = """You repair failed Python solutions. Return exactly one JSON object with:
diagnosis, problem, solution, tests, category, skill, difficulty, task_type, complexity.
The corrected solution must address the supplied failure and remain self-contained."""


def seed_prompt(seed: dict) -> str:
    return f"""Create a NEW training example inspired by this seed, not a paraphrase.
Target category: {seed.get('category','python')}
Skill: {seed.get('skill','general')}
Difficulty: {seed.get('difficulty',1)}
Task type: {seed.get('task_type','implementation')}
Seed prompt:\n{seed.get('prompt','')}\nSeed answer:\n{seed.get('response','')}"""


def repair_prompt(row: dict) -> str:
    return f"""Problem:\n{row.get('prompt','')}\n\nStudent response:\n{row.get('student_response','')}\n\nFailure:\n{row.get('test_failure','')}\nRepair it and provide stronger tests."""
