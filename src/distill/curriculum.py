from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import yaml


DEFAULT_TASK_TYPES = (
    "implementation",
    "explanation",
    "debugging",
    "repair",
    "optimization",
    "test_generation",
    "edge_case_analysis",
    "code_reading",
    "output_prediction",
    "algorithm_selection",
    "complexity_analysis",
    "alternative_solution",
    "refactoring",
)


@dataclass(frozen=True)
class CurriculumCell:
    domain: str
    stage_id: str
    stage_name: str
    topic_id: str
    topic_name: str
    level: int
    level_name: str
    task_type: str

    @property
    def key(self) -> str:
        return f"{self.topic_id}/L{self.level}/{self.task_type}"


def load_curriculum(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "domain" not in data or "stages" not in data:
        raise ValueError(f"invalid curriculum file: {path}")
    return data


def iter_cells(
    curriculum: dict,
    task_types: Iterable[str] = DEFAULT_TASK_TYPES,
    *,
    min_level: int | None = None,
    max_level: int | None = None,
) -> Iterable[CurriculumCell]:
    levels = {int(k): str(v) for k, v in curriculum.get("levels", {}).items()}
    for stage in curriculum["stages"]:
        for topic in stage["topics"]:
            lo = int(topic.get("min_level", 0))
            hi = int(topic.get("max_level", max(levels) if levels else 5))
            if min_level is not None:
                lo = max(lo, min_level)
            if max_level is not None:
                hi = min(hi, max_level)
            for level in range(lo, hi + 1):
                for task_type in task_types:
                    yield CurriculumCell(
                        domain=curriculum["domain"],
                        stage_id=stage["id"],
                        stage_name=stage["name"],
                        topic_id=topic["id"],
                        topic_name=topic["name"],
                        level=level,
                        level_name=levels.get(level, f"level_{level}"),
                        task_type=task_type,
                    )
