from __future__ import annotations
from src.verification.sandbox import run_python, RunResult


def compose_test_program(solution: str, tests) -> str:
    if isinstance(tests, str):
        test_code=tests
    else:
        test_code="\n".join(str(x) for x in tests)
    return solution.rstrip()+"\n\n# --- generated tests ---\n"+test_code+"\n"


def verify_solution(solution: str, tests, timeout: float=4.0) -> RunResult:
    try:
        compile(solution,"<solution>","exec")
    except SyntaxError as e:
        return RunResult(False,2,"",f"SyntaxError: {e}")
    return run_python(compose_test_program(solution,tests), timeout=timeout)
