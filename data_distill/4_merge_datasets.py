import json
import os
import glob
import random

INPUT_DIR = "."  # Folder containing all your JSON files
OUTPUT_FILE = "./final_master_dataset.json"

print("🔍 Scanning for JSON files...")
json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

if not json_files:
    print("❌ No JSON files found in the directory!")
    exit()

print(f"📂 Found {len(json_files)} files. Merging...")

master_dataset = []
for file_path in json_files:
    print(f"  Loading {os.path.basename(file_path)}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                master_dataset.extend(data)
            else:
                print(f"⚠️  Skipped {os.path.basename(file_path)}: Not a JSON list")
        except json.JSONDecodeError:
            print(f"❌ Failed to parse {os.path.basename(file_path)}")

print(f"\n🔀 SHUFFLING {len(master_dataset)} total examples...")
print("   (This prevents recency bias and catastrophic forgetting)")
random.seed(42) # Set seed for reproducible shuffling
random.shuffle(master_dataset)

print(f"💾 Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(master_dataset, f, indent=2)

print(f"✅ Success! Master dataset created with {len(master_dataset)} examples.")

# Optional: Verify the shuffle worked
print("\n📊 First 5 topics in the shuffled dataset:")
for i in range(min(5, len(master_dataset))):
    print(f"  {i+1}. {master_dataset[i].get('topic')} ({master_dataset[i].get('difficulty')})")