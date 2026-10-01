import json
from pathlib import Path

from src.distill.curriculum import CurriculumCell
from src.distill.factory import DistillationFactory
from src.distill.teachers.base import TeacherBackend, TeacherConfig


GOOD = {
    "problem": "Write a function add(a, b) that returns a + b.",
    "solution": "def add(a, b):\n    return a + b",
    "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    "explanation": "Return the arithmetic sum.",
    "time_complexity": "O(1)",
    "space_complexity": "O(1)",
}


class MockTeacher(TeacherBackend):
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.config = TeacherConfig(
            name="mock",
            backend="mock",
            model_id="mock/model",
            revision="test",
            generation={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 100},
        )

    def generate_text(self, system: str, user: str) -> str:
        return next(self.outputs)


def cell(task_type="implementation"):
    return CurriculumCell("python", "P0", "fundamentals", "P00", "syntax", 1, "beginner", task_type)


def test_factory_accepts_good_and_rejects_bad_json_and_duplicates(tmp_path: Path):
    good = json.dumps(GOOD)
    teacher = MockTeacher([good, "```json\n" + good + "\n```", good])
    factory = DistillationFactory([teacher], tmp_path, run_id="test", near_duplicate_threshold=None)
    result = factory.run([cell(), cell(), cell()], samples_per_cell=1)

    assert result.generated == 3
    assert result.accepted == 1
    assert result.rejected == 1
    assert result.duplicates == 1

    accepted = [json.loads(x) for x in (tmp_path / "accepted.jsonl").read_text().splitlines()]
    assert accepted[0]["topic_id"] == "P00"
    assert accepted[0]["fingerprints"]["prompt_sha256"]

    rejected = [json.loads(x) for x in (tmp_path / "rejected.jsonl").read_text().splitlines()]
    assert rejected[0]["reason"] == "invalid_json"

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["generated"] == 3
    assert manifest["accepted"] == 1
