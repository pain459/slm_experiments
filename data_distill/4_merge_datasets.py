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
                print(f"⚠️  Skipped {file_path}: Not a JSON list")
        except json.JSONDecodeError:
            print(f"❌ Failed to parse {file_path}")

print(f"\n🔀 Shuffling {len(master_dataset)} total examples to prevent catastrophic forgetting...")
random.shuffle(master_dataset)

print(f"💾 Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(master_dataset, f, indent=2)

print(f"✅ Success! Master dataset created with {len(master_dataset)} examples.")