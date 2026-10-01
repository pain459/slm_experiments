from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class TeacherConfig:
    name: str
    backend: str
    model_id: str
    revision: str = "main"
    enabled: bool = True
    load_in_4bit: bool = False
    generation: dict[str, Any] = field(default_factory=dict)


def load_teacher_configs(path: str | Path) -> list[TeacherConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    teachers = []
    for item in raw.get("teachers", []):
        teachers.append(TeacherConfig(
            name=item["name"],
            backend=item.get("backend", "hf"),
            model_id=item["model_id"],
            revision=item.get("revision", "main"),
            enabled=bool(item.get("enabled", True)),
            load_in_4bit=bool(item.get("load_in_4bit", False)),
            generation=dict(item.get("generation", {})),
        ))
    return teachers


class TeacherBackend(ABC):
    config: TeacherConfig

    @abstractmethod
    def generate_text(self, system: str, user: str) -> str:
        raise NotImplementedError
