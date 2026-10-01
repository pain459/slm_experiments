# Python Specialist Lab — Six-Module Repository

This repository implements the frozen six-module architecture for building a small local Python + DSA coding agent.

```text
1. Reverse distillation
       ↓
2. Independent judge + confidence
       ↓
3. Curriculum-aware stratified dataset
       ↓
4. Small English-model training + GGUF/Ollama
       ↓
5. Agentic long-horizon post-training
       ↓
6. Context virtualization + persistent memory + OpenCode
```

## Modules

### 1 — `01_distill/` + `src/distill/`
Ordered Python and DSA curriculum, multi-teacher configs, strict JSON parsing, schema validation, provenance, exact/near dedup and run manifests.

### 2 — `02_judge/` + `src/judge/`
Independent model judging, independent adversarial-test generation, execution hard gate, component scores and GOLD/SILVER/REVIEW/REJECT tiers.

### 3 — `03_curriculum/` + `src/curriculum_builder/`
Stage assignment, deterministic stratified interleaving across domain/topic/level/task/teacher/tier, manifest and dataset fingerprint.

### 4 — `04_train/` + `src/unsloth_train/`
Unsloth SFT/QLoRA entrypoint, stage-ready dataset consumption, GGUF export helper and Ollama Modelfile template. Unsloth is intentionally optional because its CUDA/PyTorch install is environment-specific.

### 5 — `05_agent/` + `src/agent/`
Tool environment, repository task runner, trajectory generation/replay and horizon classification. Agent post-training should be a second phase on top of the Python/DSA specialist.

### 6 — `06_context_proxy/` + `src/context_proxy/`
OpenAI-compatible frontend proxy in front of Ollama. It keeps raw conversation in SQLite, semantic memory in ChromaDB (with lexical fallback), symbol-level Python code index, compaction, retrieval and bounded working-context construction. OpenCode talks to this proxy, so normal sessions can extend far beyond the model's native context window.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```

For the context proxy:

```bash
pip install -e '.[proxy]'
```

For Unsloth, install a CUDA-compatible Unsloth environment separately, then use `04_train/README.md`.

## End-to-end commands

### Module 1

```bash
python -m src.distill.plan   --curriculum 01_distill/curriculum/python.yaml 01_distill/curriculum/dsa.yaml   --task-types implementation debugging optimization test_generation explanation   --samples-per-cell 1

RUN_ID=rd_v1_pilot MAX_CELLS=20 SAMPLES_PER_CELL=1 ./scripts/08_distill_module1.sh
```

### Module 2

```bash
python -m src.judge.run   --input 01_distill/outputs/rd_v1_pilot/accepted.jsonl   --output-dir 02_judge/outputs/judge_v1
```

### Module 3

```bash
python -m src.curriculum_builder.build   --inputs 02_judge/outputs/judge_v1/gold.jsonl 02_judge/outputs/judge_v1/silver.jsonl   --output-dir 03_curriculum/outputs/dataset_v1
```

### Module 4

```bash
python -m src.unsloth_train.train --config 04_train/configs/unsloth_v1.yaml
python -m src.unsloth_train.export --model artifacts/models/python_dsa_v1 --out artifacts/gguf
```

### Module 5

Use the existing repository tools/runner to generate successful trajectories, replay them from clean repository state, then post-train an agent adapter. See `05_agent/README.md`.

### Module 6

```bash
ollama serve
python -m src.context_proxy.server --repo /path/to/current/project
```

Configure OpenCode with `06_context_proxy/opencode/opencode.json.example` so it uses `http://127.0.0.1:8787/v1` rather than talking directly to Ollama.

## Pipeline wrapper

```bash
python pipeline.py distill --help
python pipeline.py judge --input ...
python pipeline.py curriculum --inputs ...
python pipeline.py train --config ...
python pipeline.py proxy --repo ...
```

## Design rules

- Fine-tuning stores Python/DSA skills and coding/agent behavior.
- The context proxy stores project-specific facts, conversation history, repository context and long-running session state.
- Executable samples that fail independent tests are rejected regardless of LLM confidence.
- Private evaluation data must never enter distillation or training.
- Generated code execution in this repo is a research starter, not a hardened security boundary. For large-scale untrusted generation, replace it with Docker/nsjail/Firecracker isolation.
