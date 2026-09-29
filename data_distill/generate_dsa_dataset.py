import ollama
import json
import os
import re
import random
from typing import List, Dict, Set

# Configuration
MODEL_NAME = "qwen2.5:7b"
OUTPUT_FILE = "comprehensive_dataset_dsa.json"
QUESTIONS_PER_TOPIC = 500
BATCH_SIZE = 3        # smaller batches: long JSON output is the #1 truncation/parse risk
NUM_PREDICT = 5000    # generous token budget for 3 detailed answers
NUM_CTX = 8192        # ollama default ctx is small; the long prompt gets truncated otherwise


# Question types and difficulties for diversity
QUESTION_TYPES = [
    "conceptual",
    "code_generation",
    "debugging",
    "comparison",
    "trace_the_execution",
    "complexity_analysis",
    "optimization",
    "edge_cases",
    "real_world_application",
    "common_mistakes",
]


DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]


TOPICS = [
        "Complexity Analysis: Big O, space/time complexity, amortized analysis",
        "Arrays & Strings: Two pointers, sliding window, prefix sums",
        "Linked Lists: Singly, doubly, circular, operations, cycle detection",
        "Stacks & Queues: Implementation, applications (parenthesis matching, BFS)",
        "Hash Tables: Hash functions, collision handling, applications",
        "Trees: Binary trees, BST, traversals (in/pre/post/level), AVL, Red-Black (conceptual)",
        "Heaps/Priority Queues: Min/max heap, heapify, applications (Dijkstra, scheduling)",
        "Graphs: Representation (adjacency list/matrix), BFS, DFS, topological sort",
        "Graph Algorithms: Dijkstra, Bellman-Ford, Floyd-Warshall, Kruskal, Prim",
        "Recursion: Base cases, recursive tree, memoization",
        "Dynamic Programming: Memoization, tabulation, classic problems (knapsack, LCS, LIS)",
        "Greedy Algorithms: Activity selection, Huffman coding, fractional knapsack",
        "Divide & Conquer: Merge sort, quick sort, binary search variations",
        "Backtracking: N-queens, sudoku solver, subset sum",
        "Trie: Prefix tree, autocomplete, word search",
        "Segment Trees & Fenwick Trees: Range queries, point updates",
        "Disjoint Set (Union-Find): Path compression, union by rank",
        "String Algorithms: KMP, Rabin-Karp, Z-algorithm",
        "Bit Manipulation: Bitwise operators, bit masks, bit tricks",
]


try:
    import json_repair
except ImportError:
    json_repair = None


def clean_json_response(response_text: str) -> str:
    """Strips markdown formatting and fixes common JSON generation errors."""
    text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text


def generate_diverse_questions(topic: str, existing_questions: Set[str], questions_needed: int) -> List[Dict]:
    """Generates diverse DSA questions for a topic until questions_needed are produced."""

    # Sample a diverse mix of question types and difficulties
    combinations = []
    for q_type in QUESTION_TYPES:
        for difficulty in DIFFICULTY_LEVELS:
            combinations.append((q_type, difficulty))

    # Shuffle to ensure random distribution
    random.shuffle(combinations)

    # Cycle through the shuffled pairs so we can exceed the 40 unique combos
    selected_combinations = [combinations[i % len(combinations)] for i in range(questions_needed)]

    all_questions = []

    for i in range(0, len(selected_combinations), BATCH_SIZE):
        batch = selected_combinations[i:i + BATCH_SIZE]

        prompt = f"""You are an expert in data structures and algorithms, creating DSA training data for the topic: "{topic}".

Generate exactly {len(batch)} diverse, high-quality questions. Each question must be UNIQUE and cover a different aspect of the topic.

For each question, specify:
- Type: {', '.join([b[0] for b in batch])}
- Difficulty: {', '.join([b[1] for b in batch])}

You MUST output ONLY valid JSON as an array of objects. Do not include any text outside the JSON.

Each object must follow this exact schema:
{{
  "question_type": "one of: {', '.join(QUESTION_TYPES)}",
  "difficulty": "one of: {', '.join(DIFFICULTY_LEVELS)}",
  "instruction": "A clear, self-contained question (2-3 sentences max). For coding tasks, state the exact function signature and expected input/output. For trace_the_execution, provide the concrete input values.",
  "input": "",
  "output": {{
    "definition": "Brief 1-2 sentence definition if applicable, otherwise empty string",
    "explanation": "Detailed 2-4 paragraph explanation: how it works, why, and the time/space complexity reasoning",
    "examples": [
      {{
        "code": "Complete, runnable Python code. For implementation tasks include the function plus a usage demo. For tracing include the step-by-step intermediate states in comments.",
        "description": "What this example shows, including expected output for the given input"
      }}
    ],
    "common_mistakes": ["Mistake 1", "Mistake 2"],
    "best_practices": ["Practice 1", "Practice 2"]
  }}
}}

CRITICAL REQUIREMENTS:
1. Questions must be DIVERSE - don't repeat similar questions or reuse the same example inputs
2. Code must be 100% syntactically correct, PEP 8 compliant, and RUNNABLE as written
3. Explanations must be detailed and educational (150-300 words) and always state time and space complexity where applicable
4. Match the difficulty level:
   - beginner: basic operations and traversal on one structure, simple applications
   - intermediate: classic patterns (two pointers, hashing, binary search, basic recursion), moderate complexity
   - advanced: graphs, DP, greedy, heaps, non-trivial edge cases, complexity trade-offs
   - expert: amortized analysis, advanced structures (segment trees, union-find, KMP), optimal approaches
5. For debugging questions: put BROKEN code in the instruction, describe the failing symptom, and give the corrected code in the output examples
6. For comparison questions: compare head-to-head (e.g., Kruskal vs Prim), state when each wins, with complexity for both
7. For trace_the_execution: use a small concrete input (8 elements or fewer) and show every iteration/pass in the explanation
8. For optimization questions: show the naive solution first, then the improved one, with complexity before/after

Return ONLY the JSON array now:
"""

        raw_text = ""
        try:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={
                    "temperature": 0.4,
                    "num_predict": NUM_PREDICT,
                    "num_ctx": NUM_CTX,
                }
            )

            raw_text = response['response']
            clean_text = clean_json_response(raw_text)

            try:
                questions_batch = json.loads(clean_text, strict=False)
            except json.JSONDecodeError:
                # If standard relaxed parsing still fails, try to repair it
                if json_repair is None:
                    print("⚠️  JSON failed to parse. Run 'pip install json-repair' for robust parsing.")
                    print(f"Error in batch {i // BATCH_SIZE + 1}. Raw snippet: {clean_text[:300]}...")
                    continue
                questions_batch = json_repair.loads(clean_text)

            if not isinstance(questions_batch, list):
                print(f"⚠️  Expected array, got {type(questions_batch).__name__}")
                continue

            required_keys = ["question_type", "difficulty", "instruction", "input", "output"]
            output_keys = ["definition", "explanation", "examples", "common_mistakes", "best_practices"]

            kept = 0
            for q in questions_batch:
                out = q.get("output") if isinstance(q, dict) else None
                if not (isinstance(q, dict) and isinstance(out, dict) and isinstance(q.get("instruction"), str)):
                    print("⚠️  Malformed question object, skipping")
                    continue

                if all(k in q for k in required_keys) and all(k in out for k in output_keys):
                    # Add topic metadata
                    q["topic"] = topic

                    # Check for duplicate instructions
                    instruction = q["instruction"]
                    if instruction not in existing_questions:
                        all_questions.append(q)
                        existing_questions.add(instruction)
                        kept += 1
                    else:
                        print(f"⚠️  Duplicate question skipped: {instruction[:50]}...")
                else:
                    print("⚠️  Question missing required keys, skipping")

            print(f"  ✅ Kept {kept}/{len(questions_batch)} questions (batch {i // BATCH_SIZE + 1})")

        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error in batch {i // BATCH_SIZE + 1}: {e}")
            print(f"Raw output snippet: {raw_text[:300]}...")
        except Exception as e:
            print(f"❌ Unexpected error in batch {i // BATCH_SIZE + 1}: {e}")

    return all_questions


def main():
    print(f"🚀 Starting comprehensive DSA data generation using {MODEL_NAME}...")
    print(f"📊 Target: {QUESTIONS_PER_TOPIC} questions per topic × {len(TOPICS)} topics = {QUESTIONS_PER_TOPIC * len(TOPICS)} total questions\n")

    # Load existing data
    existing_data = []
    existing_questions: Set[str] = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_data = json.load(f)
            existing_questions = {
                item["instruction"] for item in existing_data
                if isinstance(item, dict) and isinstance(item.get("instruction"), str)
            }
            print(f"📂 Loaded {len(existing_data)} existing questions")
        except Exception:
            print("⚠️  Could not read existing file, starting fresh.")

    # Track progress
    topic_stats = {}
    for topic in TOPICS:
        topic_stats[topic] = len([q for q in existing_data if q.get("topic") == topic])

    print(f"\n📈 Current progress:")
    for topic, count in topic_stats.items():
        print(f"  {topic}: {count}/{QUESTIONS_PER_TOPIC} questions")

    # Generate for each topic
    for topic in TOPICS:
        current_count = topic_stats[topic]

        if current_count >= QUESTIONS_PER_TOPIC:
            print(f"\n✅ {topic}: Already has {current_count} questions, skipping")
            continue

        questions_needed = QUESTIONS_PER_TOPIC - current_count
        print(f"\n{'=' * 60}")
        print(f"📝 Generating {questions_needed} questions for: {topic}")
        print(f"{'=' * 60}")

        new_questions = generate_diverse_questions(topic, existing_questions, questions_needed)

        if new_questions:
            existing_data.extend(new_questions)

            # Save incrementally
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(existing_data, f, indent=2)

            topic_stats[topic] += len(new_questions)
            print(f"\n✅ {topic}: Added {len(new_questions)} questions (Total: {topic_stats[topic]}/{QUESTIONS_PER_TOPIC})")
        else:
            print(f"\n⚠️  {topic}: No valid questions generated")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"🎉 GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total questions: {len(existing_data)}")
    print(f"\nBreakdown by topic:")
    for topic, count in topic_stats.items():
        print(f"  {topic}: {count}")

    # Breakdown by difficulty
    difficulty_counts = {}
    for q in existing_data:
        diff = q.get("difficulty", "unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    print(f"\nBreakdown by difficulty:")
    for diff in DIFFICULTY_LEVELS:
        print(f"  {diff}: {difficulty_counts.get(diff, 0)}")

    # Breakdown by question type
    type_counts = {}
    for q in existing_data:
        qtype = q.get("question_type", "unknown")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    print(f"\nBreakdown by question type:")
    for qtype in QUESTION_TYPES:
        print(f"  {qtype}: {type_counts.get(qtype, 0)}")


if __name__ == "__main__":
    main()
