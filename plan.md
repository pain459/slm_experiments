# Final Blueprint: Small Python + DSA Coding Agent

## 0. Final objective

Build a small model that:

- starts primarily with English capability
- learns Python in a curriculum from basics to advanced
- learns DSA from fundamentals to Olympiad-style problems
- gains debugging, optimization, testing, and repair skills
- develops long-horizon agentic coding behavior
- runs locally through Ollama
- works with OpenCode
- uses a frontend/context proxy so the user experiences effectively unbounded working context despite the model having a finite native context window
- stores historical conversation, repository context, decisions, code, summaries, and session state externally

The architecture is divided into six modules:

```text
MODULE 1
Reverse-distillation data factory
        ↓
MODULE 2
Independent judging and confidence scoring
        ↓
MODULE 3
Curriculum-aware stratified dataset construction
        ↓
MODULE 4
Small-model training + GGUF/Ollama export
        ↓
MODULE 5
Agentic coding / long-horizon post-training
        ↓
MODULE 6
Context virtualization + persistent memory + OpenCode runtime
```

---

# 1. MODULE 1 — Reverse-Distillation Data Factory

## Goal

Generate a comprehensive Python + DSA training corpus from strong teacher models.

The system must:

- cover the complete curriculum
- generate multiple task forms
- use multiple teachers where practical
- reject malformed outputs immediately
- deduplicate during generation
- store provenance for every example
- produce reproducible JSONL datasets

---

## 1.1 Teacher strategy

Use teacher models in roles.

### Primary coding teacher

Use a strong Qwen Coder-class model.

Example role:

```text
Primary teacher:
Qwen3-Coder-class model
```

### Secondary teacher

Use another capable coding model from a different model family.

Purpose:

```text
diversify reasoning patterns
reduce single-teacher bias
generate alternative solutions
```

### Hard-problem teacher

Optional larger model only for:

```text
advanced DSA
Olympiad problems
complex debugging
repository-scale problems
failed examples
```

We do not need the largest teacher for every sample.

---

# 1.2 Python curriculum

## Stage P0 — Fundamentals

```text
P00 Syntax and execution
P01 Variables
P02 Primitive types
P03 Operators
P04 Conditions
P05 Loops
P06 Functions
P07 Scope
P08 Basic input/output
```

## Stage P1 — Core containers

```text
P09 Strings
P10 Lists
P11 Tuples
P12 Dicts
P13 Sets
P14 Slicing
P15 Comprehensions
P16 Unpacking
```

## Stage P2 — Program structure

```text
P17 Files
P18 Exceptions
P19 Modules
P20 Packages
P21 Classes
P22 Inheritance
P23 Dataclasses
P24 Properties
```

## Stage P3 — Advanced language features

```text
P25 Iterators
P26 Generators
P27 Decorators
P28 Closures
P29 Context managers
P30 Functional patterns
P31 Type hints
P32 Protocol-style design
```

## Stage P4 — Standard library

```text
P33 collections
P34 itertools
P35 functools
P36 pathlib
P37 regex
P38 datetime
P39 serialization
P40 subprocess concepts
P41 logging
```

## Stage P5 — Engineering

```text
P42 Testing
P43 Debugging
P44 Profiling
P45 Packaging
P46 CLI
P47 APIs
P48 Databases
P49 Configuration
P50 Dependency management
```

## Stage P6 — Parallelism

```text
P51 threading
P52 multiprocessing
P53 async / await
P54 queues
P55 synchronization
```

## Stage P7 — Advanced internals

```text
P56 descriptors
P57 metaclasses
P58 object model
P59 memory behavior
P60 bytecode concepts
P61 performance
P62 advanced typing
```

---

# 1.3 DSA curriculum

## D0 — Foundations

```text
D00 Complexity
D01 Arrays
D02 Strings
D03 Hash maps
D04 Sets
D05 Sorting
D06 Binary search
```

## D1 — Linear structures

```text
D07 Stack
D08 Queue
D09 Deque
D10 Linked list
D11 Recursion
```

## D2 — Trees

```text
D12 Binary trees
D13 BST
D14 Heaps
D15 Tries
```

## D3 — Graphs

```text
D16 Graph representation
D17 BFS
D18 DFS
D19 Topological sort
D20 Union-find
D21 Shortest paths
D22 MST
```

## D4 — Common problem-solving patterns

```text
D23 Two pointers
D24 Sliding window
D25 Prefix sums
D26 Difference arrays
D27 Greedy
D28 Backtracking
```

## D5 — Dynamic programming

```text
D29 DP fundamentals
D30 1D DP
D31 2D DP
D32 Knapsack
D33 subsequence DP
D34 interval DP
D35 tree DP
D36 digit DP
D37 bitmask DP
```

## D6 — Advanced structures

```text
D38 Fenwick tree
D39 Segment tree
D40 Sparse table
D41 disjoint structures
```

## D7 — Strings

```text
D42 KMP
D43 Z algorithm
D44 rolling hash
D45 suffix structures
D46 trie applications
```

## D8 — Mathematics

```text
D47 Number theory
D48 modular arithmetic
D49 combinatorics
D50 primes
D51 gcd/lcm
D52 matrix techniques
```

## D9 — Advanced algorithms

```text
D53 advanced graph algorithms
D54 tree algorithms
D55 advanced DP
D56 computational geometry
D57 randomized algorithms
D58 offline algorithms
```

## D10 — Competition

```text
D59 mixed hard problems
D60 competitive programming
D61 Olympiad-style reasoning
```

---

# 1.4 Difficulty levels

Every topic has explicit levels:

```text
L0 Concept
L1 Beginner
L2 Intermediate
L3 Advanced
L4 Hard
L5 Olympiad / adversarial
```

---

# 1.5 Task types

Every topic should produce multiple task forms:

```text
implementation
explanation
debugging
repair
optimization
test generation
edge-case analysis
code reading
output prediction
algorithm selection
complexity analysis
alternative solution
refactoring
```

A sample is therefore addressed like:

```text
D29 / L2 / implementation
P43 / L3 / debugging
D21 / L4 / optimization
```

---

# 1.6 Generation format

Every teacher response must conform to a schema.

Example:

```json
{
  "id": "qwen_d29_l2_impl_000001",
  "domain": "dsa",
  "topic_id": "D29",
  "level": 2,
  "task_type": "implementation",

  "teacher_model": "...",
  "teacher_revision": "...",

  "prompt": "...",
  "answer": "...",

  "tests": [],
  "explanation": "...",

  "metadata": {
    "time_complexity": "...",
    "space_complexity": "..."
  }
}
```

---

# 1.7 Generation pipeline

```text
curriculum cell
      ↓
teacher generation
      ↓
JSON parse
      ↓
schema validation
      ↓
normalization
      ↓
exact dedup
      ↓
near-dedup
      ↓
accepted candidate dataset
```

Rejected categories:

```text
invalid_json
invalid_schema
duplicate
near_duplicate
missing_fields
unsupported_language
empty_answer
```

---

# 1.8 Deduplication

Use at least:

```text
normalized prompt hash
normalized code hash
prompt + answer hash
```

Then optional:

```text
MinHash
embedding similarity
AST-normalized code similarity
```

---

# 1.9 Provenance

Every sample stores:

```text
teacher
teacher revision
prompt template version
temperature
top_p
seed topic
level
task type
generation timestamp
generation attempt
source teacher
```

---

# 1.10 Module 1 output

```text
01_distill/
    outputs/
        teacher_a/
            accepted.jsonl
            rejected_json.jsonl
            duplicates.jsonl

        teacher_b/
            accepted.jsonl
            rejected_json.jsonl
            duplicates.jsonl

        merged_candidates.jsonl
```

---

# 1.11 Module 1 success criteria

```text
JSON validity              >98%
schema validity            >98%
exact duplicates           <2%
curriculum coverage        100%
all samples have provenance
all required topic/level buckets represented
```

---

# 2. MODULE 2 — Independent Judge + Confidence Scoring

## Goal

Run every generated sample through an independent sanity check.

This module must combine:

```text
execution verification
independent LLM judging
clarity checks
test quality checks
curriculum-fit checks
```

---

# 2.1 Judge independence

Prefer a judge model from a different family than the primary teacher.

This reduces correlated errors.

---

# 2.2 Executable examples

For coding tasks:

```text
solution
   ↓
syntax check
   ↓
sandbox execution
   ↓
unit tests
   ↓
hidden / independent tests
```

If execution fails:

```text
automatic reject
```

The model judge cannot override failed execution.

---

# 2.3 Independent test generation

Do not rely only on teacher-provided tests.

Pattern:

```text
Teacher A
→ problem + answer

Judge/Test model
→ independent tests

Verifier
→ execute answer against independent tests
```

This prevents self-confirming mistakes.

---

# 2.4 Confidence score

For executable tasks:

```text
execution             50%
judge correctness     20%
problem clarity       10%
test quality          10%
curriculum fit        10%
```

For explanation/non-executable tasks:

```text
correctness           50%
completeness          20%
concept consistency   15%
curriculum fit        15%
```

Store every sub-score.

---

# 2.5 Confidence tiers

```text
>= 0.90    GOLD
0.80–0.899 SILVER
0.70–0.799 REVIEW
< 0.70     REJECT
```

Executable failure is always:

```text
REJECT
```

---

# 2.6 Output

```text
02_judge/
    outputs/
        gold.jsonl
        silver.jsonl
        review.jsonl
        rejected.jsonl
        score_report.json
```

---

# 2.7 Success criteria

```text
100% accepted candidates scored
100% executable samples executed
independent tests used
confidence distribution generated
rejection reasons recorded
judge model and version recorded
```

---

# 3. MODULE 3 — Stratified Curriculum Dataset Builder

## Goal

Build one training dataset without destroying curriculum structure.

Do not globally random-shuffle everything.

Use staged curriculum + stratified interleaving.

---

# 3.1 Curriculum stages

## Stage A

```text
Python fundamentals
basic containers
basic DSA
complexity
linear structures
```

## Stage B

```text
OOP
exceptions
modules
trees
graphs
common patterns
```

## Stage C

```text
standard library
testing
engineering
DP
advanced structures
```

## Stage D

```text
advanced Python
advanced graph algorithms
string algorithms
math
competition
Olympiad
```

---

# 3.2 Stratification keys

Shuffle across:

```text
domain
topic
level
task type
teacher
confidence tier
```

---

# 3.3 Sampling weights

Example:

```text
gold      1.0
silver    0.6
```

Difficulty weighting:

```text
L0  0.8
L1  1.0
L2  1.1
L3  1.2
L4  1.0
L5  0.7
```

Hard/Olympiad examples must not dominate early training.

---

# 3.4 Reproducibility

Every shuffle uses:

```text
fixed random seed
dataset manifest
source hashes
```

---

# 3.5 Output

```text
03_curriculum/
    outputs/
        stage_a.jsonl
        stage_b.jsonl
        stage_c.jsonl
        stage_d.jsonl
        train_full.jsonl
        manifest.json
```

Example manifest:

```json
{
  "P01": {
    "L0": 400,
    "L1": 850,
    "L2": 700
  },
  "D29": {
    "L1": 500,
    "L2": 750,
    "L3": 600
  }
}
```

---

# 3.6 Success criteria

```text
no train/eval leakage
no duplicate leakage
every curriculum bucket reaches minimum count
balanced source diversity
reproducible shuffle
no extreme topic dominance
```

---

# 4. MODULE 4 — Small English Model Training

## Goal

Teach a small English-capable model Python + DSA.

Recommended first proof size:

```text
0.5B–1.5B
```

Potential later size:

```text
3B
```

---

# 4.1 Training framework

Use:

```text
Unsloth
Transformers
PEFT
LoRA / QLoRA
```

Start with LoRA/QLoRA for iteration speed.

---

# 4.2 Training progression

```text
English base
   ↓
Stage A
   ↓
checkpoint_A
   ↓
Stage B
   ↓
checkpoint_B
   ↓
Stage C
   ↓
checkpoint_C
   ↓
Stage D
   ↓
python_dsa_v1
```

---

# 4.3 Evaluate after every stage

Do not train all stages blindly.

Track:

```text
Python basic
Python intermediate
Python advanced
DSA basic
DSA medium
DSA hard
debugging
optimization
testing
```

Example:

| Model | Py Basic | Py Adv | DSA Basic | DSA Hard | Debug |
|---|---:|---:|---:|---:|---:|
| Base | | | | | |
| Stage A | | | | | |
| Stage B | | | | | |
| Stage C | | | | | |
| Stage D | | | | | |

---

# 4.4 Output formats

Produce:

```text
Hugging Face checkpoint
merged checkpoint
GGUF F16
GGUF Q8_0
GGUF Q4_K_M
Ollama Modelfile
```

---

# 4.5 Ollama pipeline

```text
HF model
   ↓
merge adapter
   ↓
GGUF conversion
   ↓
quantization
   ↓
Modelfile
   ↓
ollama create
```

---

# 4.6 Quantization evaluation

Test:

```text
HF
F16 GGUF
Q8_0
Q4_K_M
```

against the same benchmark.

Measure regression.

---

# 4.7 Module 4 success criteria

```text
trained model beats English baseline
knowledge progression visible across stages
HF model loads
GGUF loads
Ollama runs
quantization loss measured
Q4 build retains acceptable performance
```

---

# 5. MODULE 5 — Agentic Coding + Long-Horizon Thinking

## Goal

Teach the trained model how to work over long coding sessions.

Module 4 teaches:

```text
coding knowledge
```

Module 5 teaches:

```text
planning
tool use
debugging loops
state management
verification
termination
```

---

# 5.1 Core tools

Initially:

```text
read_file
write_file
edit_file
search_files
run_shell
run_python
run_tests
git_diff
```

---

# 5.2 Trajectory format

```text
user task
   ↓
inspect
   ↓
tool call
   ↓
observation
   ↓
plan/update
   ↓
edit
   ↓
execute
   ↓
failure
   ↓
repair
   ↓
test
   ↓
verify
   ↓
finish
```

---

# 5.3 Long-horizon buckets

```text
H1  1–3 tool calls
H2  4–8
H3  9–20
H4  20–50
H5  repository-scale
```

Train progressively.

---

# 5.4 Agent training data

Generate from strong coding teachers inside real/synthetic repositories.

Retain only:

```text
successful trajectories
replayable trajectories
useful failed→repaired trajectories
```

Reject:

```text
invalid tool calls
unrecoverable runs
irreproducible outcomes
unnecessary destructive edits
```

---

# 5.5 Train second adapter

```text
python_dsa_v1
      +
agent trajectory data
      ↓
python_agent_v1
```

Keep the agent post-training separate from core knowledge training.

---

# 5.6 Agent evaluation

Use unseen repositories.

Measure:

```text
task completion
tests passed
tool validity
tool efficiency
error recovery
unnecessary changes
termination correctness
long-horizon success
```

---

# 5.7 Success criteria

```text
agent_v1 > python_dsa_v1 on repository tasks
tool validity >95%
successful replay of training trajectories
minimal regression on Python/DSA benchmark
long-horizon completion improves
```

---

# 6. MODULE 6 — Context Virtualization & Persistent Memory Proxy

## Goal

Make a finite-context local model behave as if it has effectively unlimited application-level context.

The user/frontend should never normally see:

```text
context exceeded
conversation too long
```

The proxy manages context automatically.

---

# 6.1 Architecture

```text
OpenCode / frontend
        │
        ▼
┌──────────────────────────────┐
│ Context Virtualization Proxy │
│                              │
│ raw history                  │
│ summaries                    │
│ vector retrieval             │
│ structured state             │
│ repository index             │
│ context planner              │
│ context packer               │
└──────────────┬───────────────┘
               │
         <= native window
               │
               ▼
           Ollama
               │
               ▼
       python_agent_v1
               │
               ▼
            response
               │
               ▼
      proxy stores updates
```

---

# 6.2 Memory layers

Use multiple stores.

## Raw history

SQLite/Postgres:

```text
sessions
messages
tool calls
timestamps
token counts
files touched
```

This is the source of truth.

## Vector memory

ChromaDB initially.

Possible later alternatives:

```text
Qdrant
Weaviate
pgvector
```

Store:

```text
conversation chunks
decisions
errors
solutions
summaries
documentation
code explanations
```

## Structured state

Example:

```json
{
  "goal": "...",
  "current_plan": [],
  "files_seen": [],
  "files_modified": [],
  "open_questions": [],
  "failed_attempts": [],
  "important_decisions": [],
  "current_error": null,
  "remaining_work": []
}
```

## Code index

Use:

```text
AST / tree-sitter
symbol index
imports
references
functions
classes
tests
file summaries
```

## Lexical search

Use:

```text
ripgrep
BM25
```

---

# 6.3 Hybrid retrieval

```text
user message
     ↓
query planner
     ↓
┌───────────────┬───────────────┬───────────────┐
│ vector search │ lexical search│ symbol search │
└───────────────┴───────────────┴───────────────┘
                     ↓
                  reranker
                     ↓
               context packer
```

---

# 6.4 Context virtualization

Suppose native context is ~32K.

Example budget:

```text
System / behavior            2K
Current request              2K
Recent raw messages          4K
Long-term summary            3K
Retrieved conversation       4K
Relevant source code         8K
Tool observations            3K
Output reserve               6K
───────────────────────────────
Total                       32K
```

This budget is dynamic.

Coding task:

```text
more source code
less old conversation
```

Architecture discussion:

```text
more decisions/history
less raw code
```

---

# 6.5 Automatic compaction

When history grows:

```text
old raw history
      ↓
summary
      ↓
structured memory extraction
      ↓
vector index
      ↓
archive original messages
```

Nothing is lost.

The model receives:

```text
recent raw turns
+
persistent summaries
+
retrieved older messages
+
relevant code
+
current state
```

---

# 6.6 Memory classification

Store memory categories:

```text
decision
requirement
bug
solution
failed_attempt
architecture
TODO
test_result
file_context
constraint
```

This improves retrieval dramatically.

---

# 6.7 Long-session behavior

A 300K-token coding session might be compressed into:

```text
goal
current plan
completed work
files changed
current error
latest tests
critical decisions
retrieved old context
recent turns
```

Only the best working set is sent to the local model.

---

# 6.8 Transparent retrieval

Normal retrieval happens automatically.

The model does not need to ask:

```text
search_memory
```

for ordinary context recovery.

Pattern:

```text
user
↓
proxy retrieves context
↓
model receives augmented prompt
```

Explicit deep searches remain available to the agent when needed.

---

# 6.9 OpenCode integration

OpenCode talks to the proxy rather than directly to Ollama.

```text
OpenCode
   ↓
OpenAI-compatible proxy endpoint
   ↓
context virtualization
   ↓
Ollama
   ↓
local agent model
```

The proxy should expose OpenAI-compatible endpoints such as:

```text
/v1/chat/completions
/v1/models
```

---

# 6.10 Module 6 success criteria

```text
sessions can exceed native model context
no normal context-overflow failure exposed to user
relevant old facts can be recovered
repository context retrieval works
context packing stays within model limit
long coding sessions preserve task state
OpenCode works through proxy
Ollama remains interchangeable behind proxy
```

---

# 7. Final Repository Layout

```text
python-specialist/
│
├── 01_distill/
│   ├── curriculum/
│   │   ├── python.yaml
│   │   └── dsa.yaml
│   ├── teachers.yaml
│   ├── prompts/
│   ├── generators/
│   ├── schemas/
│   ├── dedup/
│   └── outputs/
│
├── 02_judge/
│   ├── judges.yaml
│   ├── execution/
│   ├── test_generation/
│   ├── scoring/
│   └── outputs/
│
├── 03_curriculum/
│   ├── stratify.py
│   ├── shuffle.py
│   ├── balance.py
│   ├── validation.py
│   └── outputs/
│
├── 04_train/
│   ├── unsloth/
│   ├── configs/
│   ├── evaluation/
│   ├── export/
│   ├── gguf/
│   └── ollama/
│
├── 05_agent/
│   ├── tools/
│   ├── environments/
│   ├── repositories/
│   ├── trajectory_generation/
│   ├── replay/
│   ├── training/
│   └── evaluation/
│
├── 06_context_proxy/
│   ├── api/
│   ├── conversations/
│   ├── memory/
│   ├── code_index/
│   ├── retrieval/
│   ├── reranking/
│   ├── context/
│   ├── providers/
│   ├── opencode/
│   └── tests/
│
├── artifacts/
│   ├── datasets/
│   ├── models/
│   ├── gguf/
│   ├── reports/
│   └── manifests/
│
├── pipeline.py
├── Makefile
└── README.md
```

---

# 8. One-command workflow

Eventually:

```bash
make distill
make judge
make curriculum
make train
make agent
make proxy
```

or:

```bash
python pipeline.py distill
python pipeline.py judge
python pipeline.py curriculum
python pipeline.py train
python pipeline.py agent
python pipeline.py proxy
```

---

# 9. Global Evaluation Suite

Keep one completely isolated benchmark.

Categories:

```text
Python fundamentals
Python advanced
DSA easy
DSA medium
DSA hard
Olympiad
debugging
optimization
test generation
repository tasks
tool use
long-horizon tasks
memory retrieval
context continuation
```

Run it after every major model checkpoint.

---

# 10. Dataset Lifecycle

```text
teacher generation
      ↓
schema validation
      ↓
deduplication
      ↓
judge scoring
      ↓
execution verification
      ↓
confidence tiering
      ↓
curriculum balancing
      ↓
stratified shuffle
      ↓
training
```

---

# 11. Model Lifecycle

```text
English base model
      ↓
Python/DSA curriculum training
      ↓
python_dsa_v1
      ↓
agent post-training
      ↓
python_agent_v1
      ↓
GGUF
      ↓
Ollama
      ↓
Context virtualization proxy
      ↓
OpenCode
```

---

# 12. Information Placement Rule

This is critical.

Use fine-tuning for:

```text
Python knowledge
algorithms
coding patterns
debugging behavior
planning
tool behavior
agent workflow
```

Use context virtualization / retrieval for:

```text
conversation history
repository contents
project-specific architecture
old decisions
documentation
current task state
large file sets
long-session memory
```

Do not use RAG to compensate for weak Python knowledge.

Do not use fine-tuning to memorize project repositories.

---

# 13. Required Quality Gates

## Module 1

```text
JSON >98%
coverage 100%
duplicates <2%
```

## Module 2

```text
all samples scored
all executable samples tested
failed tests rejected
```

## Module 3

```text
no leakage
balanced curriculum
reproducible shuffle
```

## Module 4

```text
student > baseline
Ollama works
quantization evaluated
```

## Module 5

```text
agent success improves
tool validity >95%
minimal core-skill regression
```

## Module 6

```text
long sessions survive beyond native context
relevant historical state recoverable
OpenCode integration works
context stays under native limit
```

---

# 14. Build Order

We should implement in exactly this order:

```text
1. Curriculum definitions
2. Teacher abstraction
3. Structured generation
4. JSON/schema rejection
5. Deduplication
6. Independent judge
7. Execution verifier
8. Confidence scoring
9. Stratified curriculum builder
10. Evaluation benchmark
11. Small Unsloth training
12. GGUF/Ollama export
13. Tool environment
14. Agent trajectory generation
15. Agent post-training
16. Conversation store
17. Vector/semantic memory
18. Code index
19. Context planner
20. Context compaction
21. OpenAI-compatible proxy
22. OpenCode integration
```

The dependency direction should never be reversed.

---

# 15. Final Product Architecture

```text
                     TEACHER MODELS
                           │
                           ▼
                 Reverse Distillation
                           │
                           ▼
                    Independent Judge
                           │
                           ▼
                Curriculum Data Builder
                           │
                           ▼
                   Small English Model
                           │
                           ▼
                   Python/DSA Training
                           │
                           ▼
                    Python Specialist
                           │
                           ▼
                   Agent Post-Training
                           │
                           ▼
                     Coding Agent
                           │
                           ▼
              GGUF → Ollama Runtime
                           │
                           ▼
       ┌────────────────────────────────┐
       │ Context Virtualization Proxy   │
       │                                │
       │ conversation memory            │
       │ vector retrieval               │
       │ code indexing                  │
       │ structured project state       │
       │ summarization / compaction     │
       │ context planner                │
       └───────────────┬────────────────┘
                       │
                       ▼
                    OpenCode
```

---

# Final principle

The project has three fundamentally different learning systems:

```text
DISTILLATION
teaches what the model knows

AGENT POST-TRAINING
teaches how the model works

CONTEXT VIRTUALIZATION
controls what the model can remember and access
```

Keeping those three concerns separate is the central architectural rule for the entire project.