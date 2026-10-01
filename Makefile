.PHONY: test distill judge curriculum train proxy

test:
	pytest

distill:
	MAX_CELLS=$${MAX_CELLS:-20} SAMPLES_PER_CELL=$${SAMPLES_PER_CELL:-1} ./scripts/08_distill_module1.sh

judge:
	./scripts/09_judge_module2.sh 01_distill/outputs/rd_v1_pilot/accepted.jsonl

curriculum:
	python -m src.curriculum_builder.build --inputs 02_judge/outputs/run_v1/gold.jsonl 02_judge/outputs/run_v1/silver.jsonl --output-dir 03_curriculum/outputs/dataset_v1

train:
	python -m src.unsloth_train.train --config 04_train/configs/unsloth_v1.yaml

proxy:
	python -m src.context_proxy.server
