import json
import random

DATASET_PATH = "dataset.json"
SAMPLE_SIZE = 10  # Check 10 random examples

with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)

print(f"📊 Dataset has {len(dataset)} total examples.\n")
print("="*80)

# Sample random examples
samples = random.sample(dataset, min(SAMPLE_SIZE, len(dataset)))

for i, item in enumerate(samples, 1):
    print(f"\n📝 EXAMPLE {i}/{SAMPLE_SIZE}")
    print(f"INSTRUCTION: {item['instruction'][:200]}...")
    print(f"\nOUTPUT: {item['output'][:300]}...")
    print("-"*80)

print("\n" + "="*80)
print("🔍 WHAT TO LOOK FOR:")
print("✅ GOOD: Answers say 'According to the text...', 'Based on Chapter X...', 'The excerpt states...'")
print("🚩 BAD: Answers include dates, names, or facts NOT in your PDF (e.g., 'discovered in 1890', 'as we know...')")
print("🚩 BAD: Answers are vague or generic without citing specific concepts from your book")
print("✅ GOOD: Cross-chapter Q&A explicitly says 'Step 1: From Chapter A... Step 2: Combining with Chapter B...'")