import ollama
import json
import os
import re
import random
from typing import List, Dict

# Configuration
MODEL_NAME = "qwen2.5:7b"
OUTPUT_FILE = "comprehensive_dataset.json"
QUESTIONS_PER_TOPIC = 100

# Question types and difficulties for diversity
QUESTION_TYPES = [
    "conceptual",
    "code_generation", 
    "debugging",
    "comparison",
    "what_happens_if",
    "best_practices",
    "performance",
    "edge_cases",
    "real_world_application",
    "common_mistakes"
]

DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]

TOPICS = [
        "Python syntax and semantics",
        "Variables, datatypes (int, float, str, bool, None)",
        "Operators (arithmetic, comparison, logical, assignment)",
        "Control flow: if/elif/else, for loops, while loops",
        "Functions: definition, parameters, return values, scope",
        "Strings: methods, slicing, formatting (f-strings, format(), %)",
        "Lists: creation, indexing, methods, slicing",
        "Tuples: immutability, unpacking",
        "Dictionaries: creation, access, methods",
        "Sets: creation, operations (union, intersection, difference)",
        "Type conversion and casting",
        "Input/output: print(), input(), file basics",
        "Error handling: try/except/finally, common exceptions",
        "Modules and imports: import, from...import, name"
]

def clean_json_response(response_text: str) -> str:
    """Strips markdown formatting and fixes common JSON generation errors."""
    text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text

def generate_diverse_questions(topic: str, existing_questions: List[str]) -> List[Dict]:
    """Generates multiple diverse questions for a topic."""
    
    # Sample a diverse mix of question types and difficulties
    combinations = []
    for q_type in QUESTION_TYPES:
        for difficulty in DIFFICULTY_LEVELS:
            combinations.append((q_type, difficulty))
    
    # Shuffle to ensure random distribution
    random.shuffle(combinations)
    
    # Take enough to reach QUESTIONS_PER_TOPIC
    selected_combinations = combinations[:QUESTIONS_PER_TOPIC]
    
    # Group by batches of 5 to reduce API calls
    batch_size = 5
    all_questions = []
    
    for i in range(0, len(selected_combinations), batch_size):
        batch = selected_combinations[i:i+batch_size]
        
        prompt = f"""You are an expert Python educator creating training data for the topic: "{topic}".

Generate exactly {len(batch)} diverse, high-quality questions. Each question must be UNIQUE and cover a different aspect of the topic.

For each question, specify:
- Type: {', '.join([b[0] for b in batch])}
- Difficulty: {', '.join([b[1] for b in batch])}

You MUST output ONLY valid JSON as an array of objects. Do not include any text outside the JSON.

Each object must follow this exact schema:
{{
  "question_type": "one of: conceptual, code_generation, debugging, comparison, what_happens_if, best_practices, performance, edge_cases, real_world_application, common_mistakes",
  "difficulty": "one of: beginner, intermediate, advanced, expert",
  "instruction": "A clear, natural-language question (2-3 sentences max)",
  "input": "",
  "output": {{
    "definition": "Brief 1-2 sentence definition if applicable, otherwise empty string",
    "explanation": "Detailed 2-4 paragraph explanation with step-by-step reasoning",
    "examples": [
      {{
        "code": "Complete, runnable Python code demonstrating the concept",
        "description": "What this example shows"
      }}
    ],
    "common_mistakes": ["Mistake 1", "Mistake 2"] if relevant, otherwise empty array,
    "best_practices": ["Practice 1", "Practice 2"] if relevant, otherwise empty array
  }}
}}

CRITICAL REQUIREMENTS:
1. Questions must be DIVERSE - don't repeat similar questions
2. Code examples must be 100% syntactically correct and follow PEP 8
3. Explanations must be detailed and educational (200-400 words)
4. Match the difficulty level appropriately:
   - beginner: basic concepts, simple examples
   - intermediate: practical applications, moderate complexity
   - advanced: edge cases, performance, deep understanding
   - expert: metaprogramming, optimization, production scenarios
5. For debugging questions, provide broken code in the instruction and the fix in the output
6. For comparison questions, compare two related concepts from the topic

Generate the JSON array now:
"""
        
        try:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={
                    "temperature": 0.4,
                    "num_predict": 4000
                }
            )
            
            raw_text = response['response']
            clean_text = clean_json_response(raw_text)

            try:
                questions_batch = json.loads(clean_text, strict=False)
            except json.JSONDecodeError:
                # If standard relaxed parsing still fails, try to repair it
                try:
                    import json_repair
                    questions_batch = json_repair.loads(clean_text)
                except ImportError:
                    print("⚠️  JSON failed to parse. Install 'json-repair' for bulletproof parsing.")
                    print(f"Error in batch {i//batch_size + 1}. Raw snippet: {clean_text[:200]}...")
                    continue

            if not isinstance(questions_batch, list):
                print(f"⚠️  Expected array, got {type(questions_batch)}")
                continue
            
            # # Parse JSON array
            # questions_batch = json.loads(clean_text)
            
            # if not isinstance(questions_batch, list):
            #     print(f"⚠️  Expected array, got {type(questions_batch)}")
            #     continue
            
            # Validate each question
            for q in questions_batch:
                required_keys = ["question_type", "difficulty", "instruction", "input", "output"]
                output_keys = ["definition", "explanation", "examples", "common_mistakes", "best_practices"]
                
                if all(k in q for k in required_keys) and all(k in q["output"] for k in output_keys):
                    # Add topic metadata
                    q["topic"] = topic
                    
                    # Check for duplicate instructions
                    if q["instruction"] not in existing_questions:
                        all_questions.append(q)
                        existing_questions.append(q["instruction"])
                    else:
                        print(f"⚠️  Duplicate question skipped: {q['instruction'][:50]}...")
                else:
                    print(f"⚠️  Question missing required keys, skipping")
            
            print(f"  ✅ Generated {len(questions_batch)} questions (batch {i//batch_size + 1})")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error in batch {i//batch_size + 1}: {e}")
            print(f"Raw output snippet: {raw_text[:300]}...")
        except Exception as e:
            print(f"❌ Unexpected error in batch {i//batch_size + 1}: {e}")
    
    return all_questions

def main():
    print(f"🚀 Starting comprehensive data generation using {MODEL_NAME}...")
    print(f"📊 Target: {QUESTIONS_PER_TOPIC} questions per topic × {len(TOPICS)} topics = {QUESTIONS_PER_TOPIC * len(TOPICS)} total questions\n")
    
    # Load existing data
    existing_data = []
    existing_questions = []
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_data = json.load(f)
            existing_questions = [item.get("instruction") for item in existing_data]
            print(f"📂 Loaded {len(existing_data)} existing questions")
        except:
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
        print(f"\n{'='*60}")
        print(f"📝 Generating {questions_needed} questions for: {topic}")
        print(f"{'='*60}")
        
        new_questions = generate_diverse_questions(topic, existing_questions)
        
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
    print(f"\n{'='*60}")
    print(f"🎉 GENERATION COMPLETE")
    print(f"{'='*60}")
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