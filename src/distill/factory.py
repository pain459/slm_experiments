from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.common.io import write_jsonl
from src.distill.curriculum import CurriculumCell
from src.distill.dedup import StreamingDeduper
from src.distill.prompts import SYSTEM, build_generation_prompt
from src.distill.schema import parse_strict_json_object, validate_teacher_object, validate_distilled_record
from src.distill.teachers.base import TeacherBackend


@dataclass
class FactoryResult:
    generated: int
    accepted: int
    rejected: int
    duplicates: int
    by_reason: dict[str, int]


class DistillationFactory:
    def __init__(
        self,
        teachers: list[TeacherBackend],
        output_dir: str | Path,
        *,
        run_id: str,
        seed: int = 42,
        near_duplicate_threshold: float | None = 0.94,
    ):
        if not teachers:
            raise ValueError("at least one teacher is required")
        self.teachers = teachers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.rng = random.Random(seed)
        self.seed = seed
        self.deduper = StreamingDeduper(near_duplicate_threshold)

    def _id_for(self, teacher: TeacherBackend, cell: CurriculumCell, sample_index: int) -> str:
        raw = f"{self.run_id}|{teacher.config.name}|{cell.key}|{sample_index}"
        suffix = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{teacher.config.name}_{cell.topic_id}_l{cell.level}_{cell.task_type}_{suffix}"

    def run(self, cells: Iterable[CurriculumCell], samples_per_cell: int = 1) -> FactoryResult:
        accepted: list[dict] = []
        rejected: list[dict] = []
        duplicates: list[dict] = []
        reasons = Counter()
        coverage = defaultdict(int)
        generated = 0

        for cell in cells:
            for teacher in self.teachers:
                for sample_index in range(samples_per_cell):
                    generated += 1
                    row_id = self._id_for(teacher, cell, sample_index)
                    nonce = f"{self.rng.getrandbits(64):016x}"
                    user_prompt = build_generation_prompt(cell, sample_index, nonce)
                    started = time.time()
                    try:
                        raw = teacher.generate_text(SYSTEM, user_prompt)
                    except Exception as exc:
                        reason = "teacher_error"
                        reasons[reason] += 1
                        rejected.append(self._reject(row_id, teacher, cell, reason, repr(exc), None, user_prompt))
                        continue

                    try:
                        obj = parse_strict_json_object(raw)
                    except ValueError as exc:
                        reason = "invalid_json"
                        reasons[reason] += 1
                        rejected.append(self._reject(row_id, teacher, cell, reason, str(exc), raw, user_prompt))
                        continue

                    errors = validate_teacher_object(obj)
                    if errors:
                        reason = "invalid_schema"
                        reasons[reason] += 1
                        rejected.append(self._reject(row_id, teacher, cell, reason, errors, raw, user_prompt))
                        continue

                    record = {
                        "id": row_id,
                        "domain": cell.domain,
                        "stage_id": cell.stage_id,
                        "stage_name": cell.stage_name,
                        "topic_id": cell.topic_id,
                        "topic": cell.topic_name,
                        "level": cell.level,
                        "level_name": cell.level_name,
                        "task_type": cell.task_type,
                        "prompt": obj["problem"].strip(),
                        "response": obj["solution"].strip(),
                        "tests": obj["tests"],
                        "explanation": obj["explanation"].strip(),
                        "complexity": {
                            "time": obj["time_complexity"].strip(),
                            "space": obj["space_complexity"].strip(),
                        },
                        "teacher": {
                            "name": teacher.config.name,
                            "backend": teacher.config.backend,
                            "model_id": teacher.config.model_id,
                            "revision": teacher.config.revision,
                        },
                        "generation": {
                            "temperature": teacher.config.generation.get("temperature", 0.4),
                            "top_p": teacher.config.generation.get("top_p", 0.95),
                            "max_new_tokens": teacher.config.generation.get("max_new_tokens", 1800),
                            "sample_index": sample_index,
                            "diversity_nonce": nonce,
                            "latency_seconds": round(time.time() - started, 4),
                        },
                        "provenance": {
                            "run_id": self.run_id,
                            "curriculum_key": cell.key,
                            "curriculum_version": 1,
                            "prompt_template": "distill_v1",
                            "random_seed": self.seed,
                        },
                        "verified": False,
                    }

                    record_errors = validate_distilled_record(record)
                    if record_errors:
                        reason = "invalid_record"
                        reasons[reason] += 1
                        rejected.append(self._reject(row_id, teacher, cell, reason, record_errors, raw, user_prompt))
                        continue

                    decision, fps = self.deduper.check(
                        row_id,
                        record["prompt"],
                        record["response"],
                        bucket=f"{cell.topic_id}:L{cell.level}:{cell.task_type}",
                    )
                    record["fingerprints"] = fps
                    if decision.duplicate:
                        reason = "duplicate"
                        reasons[reason] += 1
                        duplicates.append({
                            "id": row_id,
                            "matched_id": decision.matched_id,
                            "reason": decision.reason,
                            "similarity": decision.similarity,
                            "teacher": record["teacher"],
                            "curriculum_key": cell.key,
                            "prompt": record["prompt"],
                        })
                        continue

                    accepted.append(record)
                    coverage[cell.key] += 1

        write_jsonl(self.output_dir / "accepted.jsonl", accepted)
        write_jsonl(self.output_dir / "rejected.jsonl", rejected)
        write_jsonl(self.output_dir / "duplicates.jsonl", duplicates)
        report = {
            "run_id": self.run_id,
            "generated": generated,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "duplicates": len(duplicates),
            "acceptance_rate": len(accepted) / generated if generated else 0.0,
            "reasons": dict(reasons),
            "coverage": dict(sorted(coverage.items())),
            "teachers": [asdict(t.config) for t in self.teachers],
            "seed": self.seed,
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return FactoryResult(generated, len(accepted), len(rejected), len(duplicates), dict(reasons))

    @staticmethod
    def _reject(row_id, teacher, cell, reason, detail, raw, user_prompt):
        return {
            "id": row_id,
            "reason": reason,
            "detail": detail,
            "teacher": {
                "name": teacher.config.name,
                "model_id": teacher.config.model_id,
                "revision": teacher.config.revision,
            },
            "curriculum_key": cell.key,
            "generation_prompt": user_prompt,
            "raw_output": raw,
        }
