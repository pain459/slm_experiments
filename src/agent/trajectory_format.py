from __future__ import annotations

def to_sft_record(task_id: str, prompt: str, trajectory: list[dict], success: bool, tests_passed: bool) -> dict:
    return {"id":task_id,"domain":"agent","topic_id":"A00","level":0,"task_type":"agent_trajectory","prompt":prompt,"answer":"","messages":trajectory,"success":bool(success),"tests_passed":bool(tests_passed),"tool_calls":sum(1 for x in trajectory if x.get('tool_call'))}
