from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.distill.curriculum import DEFAULT_TASK_TYPES, iter_cells, load_curriculum
from src.distill.factory import DistillationFactory
from src.distill.teachers import load_teacher_configs


def build_teacher(config):
    if config.backend == "hf":
        from src.distill.teachers.hf import HuggingFaceTeacher
        return HuggingFaceTeacher(config)
    raise ValueError(f"unsupported backend: {config.backend}")


def main():
    ap = argparse.ArgumentParser(description="Module 1: curriculum-driven reverse distillation")
    ap.add_argument("--curriculum", nargs="+", required=True, help="YAML curriculum files")
    ap.add_argument("--teachers", default="01_distill/teachers.yaml")
    ap.add_argument("--teacher", action="append", help="Teacher name to enable; repeatable. Default: all enabled")
    ap.add_argument("--task-types", nargs="+", default=list(DEFAULT_TASK_TYPES))
    ap.add_argument("--samples-per-cell", type=int, default=1)
    ap.add_argument("--min-level", type=int)
    ap.add_argument("--max-level", type=int)
    ap.add_argument("--max-cells", type=int, help="Pilot/debug limit after curriculum expansion")
    ap.add_argument("--output", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-duplicate-threshold", type=float, default=0.94)
    args = ap.parse_args()

    configs = load_teacher_configs(args.teachers)
    selected_names = set(args.teacher or [])
    configs = [c for c in configs if (c.name in selected_names if selected_names else c.enabled)]
    if not configs:
        raise SystemExit("No teachers selected/enabled")

    cells = []
    for path in args.curriculum:
        curriculum = load_curriculum(path)
        cells.extend(iter_cells(
            curriculum,
            args.task_types,
            min_level=args.min_level,
            max_level=args.max_level,
        ))
    if args.max_cells is not None:
        cells = cells[:args.max_cells]

    teachers = [build_teacher(c) for c in configs]
    factory = DistillationFactory(
        teachers,
        Path(args.output),
        run_id=args.run_id,
        seed=args.seed,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    result = factory.run(cells, samples_per_cell=args.samples_per_cell)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
