from __future__ import annotations

import argparse
import json
from collections import Counter
from src.distill.curriculum import DEFAULT_TASK_TYPES, iter_cells, load_curriculum
from src.distill.teachers.base import load_teacher_configs


def main():
    ap = argparse.ArgumentParser(description="Preview reverse-distillation workload without loading models")
    ap.add_argument("--curriculum", nargs="+", required=True)
    ap.add_argument("--teachers", default="01_distill/teachers.yaml")
    ap.add_argument("--teacher", action="append")
    ap.add_argument("--task-types", nargs="+", default=list(DEFAULT_TASK_TYPES))
    ap.add_argument("--samples-per-cell", type=int, default=1)
    ap.add_argument("--min-level", type=int)
    ap.add_argument("--max-level", type=int)
    args = ap.parse_args()

    configs = load_teacher_configs(args.teachers)
    selected = set(args.teacher or [])
    configs = [c for c in configs if (c.name in selected if selected else c.enabled)]

    cells = []
    for path in args.curriculum:
        cells.extend(iter_cells(load_curriculum(path), args.task_types,
                                min_level=args.min_level, max_level=args.max_level))

    by_domain = Counter(c.domain for c in cells)
    by_level = Counter(f"L{c.level}" for c in cells)
    by_task = Counter(c.task_type for c in cells)
    planned = len(cells) * len(configs) * args.samples_per_cell
    result = {
        "enabled_teachers": [c.name for c in configs],
        "teachers": len(configs),
        "curriculum_cells": len(cells),
        "samples_per_cell": args.samples_per_cell,
        "planned_generations": planned,
        "cells_by_domain": dict(sorted(by_domain.items())),
        "cells_by_level": dict(sorted(by_level.items())),
        "cells_by_task_type": dict(sorted(by_task.items())),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
