import json

INPUT_FILE = "python_dataset.json"  # Replace with your actual filename
OUTPUT_FILE = "dataset.json"

with open(INPUT_FILE, 'r') as f:
    raw_data = json.load(f)

formatted_data = []
for item in raw_data:
    instruction = f"Question: {item['question']}\nExample: {item['example']}"
    output = f"{item['answer']}\n\nExplanation: {item['explanation']}"
    
    formatted_data.append({
        "instruction": instruction,
        "input": "",
        "output": output
    })

with open(OUTPUT_FILE, 'w') as f:
    json.dump(formatted_data, f, indent=2)

print(f"✅ Converted {len(formatted_data)} examples. Saved to {OUTPUT_FILE}")