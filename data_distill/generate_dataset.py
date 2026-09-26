import ollama
import json
import os
import re

# Configuration
MODEL_NAME = "qwen2.5:7b"
OUTPUT_FILE = "synthetic_dataset.json"
TOPICS = [
        "Python syntax and semantics",
        "Variables, datatypes (int, float, str, bool, None)"
        "Operators (arithmetic, comparison, logical, assignment)",
        "Control flow: if/elif/else, for loops, while loops",
        "Functions: definition, parameters, return values, scope",
        "Strings: methods, slicing, formatting (f-strings, format(), %)",
        "Lists: creation, indexing, methods, slicing",
        "Tuples: immutability, unpacking",
        "Dictionaries: creation, access, methods",
        "Sets\: creation, operations (union, intersection, difference)",
        "Type conversion and casting",
        "Input/output: print(), input(), file basics",
        "Error handling: try/except/finally, common exceptions",
        "Modules and imports: import, from...import, name",
]

def clean_json_response(response_text):
    """Strips markdown formatting and fixes common JSON generation errors."""
    # Remove ```json and ``` wrappers
    text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text

def generate_topic_data(topic):
    """Prompts the local LLM to generate structured data for a specific topic."""
    
    prompt = f"""You are an expert Python educator. Your task is to generate comprehensive, highly accurate training data for the topic: "{topic}".

You MUST output ONLY valid JSON matching this exact schema. Do not include any conversational text outside the JSON.

{{
  "topic": "{topic}",
  "instruction": "A clear, natural-language question a student would ask about this topic.",
  "input": "",
  "output": {{
    "definition": "A concise, 3-4 sentence definition of the concept.",
    "explanation": "A detailed, 2-3 paragraph explanation of how it works and why it is used.",
    "examples": [
      {{
        "code": "A clean, well-commented Python code snippet demonstrating the concept.",
        "description": "A brief description of what this specific example shows."
      }}
    ],
    "common_mistakes": [
      "Mistake 1: A specific, realistic error beginners make with this topic.",
      "Mistake 2: Another common pitfall."
    ],
    "best_practices": [
      "Best practice 1: A professional tip for using this correctly.",
      "Best practice 2: Another professional tip."
    ]
  }}
}}

Ensure the Python code in the examples is 100% syntactically correct and follows PEP 8 standards.
"""

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.3,  # Low temperature for strict, factual adherence
                "num_predict": 10000  # Allow enough tokens for a detailed response
            }
        )
        
        raw_text = response['response']
        clean_text = clean_json_response(raw_text)
        
        # Parse and validate JSON
        data = json.loads(clean_text)
        
        # Basic validation to ensure required keys exist
        required_keys = ["topic", "instruction", "input", "output"]
        output_keys = ["definition", "explanation", "examples", "common_mistakes", "best_practices"]
        
        if all(k in data for k in required_keys) and all(k in data["output"] for k in output_keys):
            return data
        else:
            print(f"⚠️  Validation failed for '{topic}': Missing keys.")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error for '{topic}': {e}")
        print(f"Raw output snippet: {raw_text[:200]}...")
        return None
    except Exception as e:
        print(f"❌ Unexpected error for '{topic}': {e}")
        return None

def main():
    print(f"🚀 Starting local data generation using {MODEL_NAME}...")
    
    # Load existing data if file exists (allows resuming)
    existing_data = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_data = json.load(f)
            print(f"📂 Loaded {len(existing_data)} existing entries from {OUTPUT_FILE}")
        except:
            print("⚠️  Could not read existing file, starting fresh.")

    # Filter out topics we already have
    existing_topics = {item.get("topic") for item in existing_data}
    topics_to_generate = [t for t in TOPICS if t not in existing_topics]
    
    print(f"📝 Generating data for {len(topics_to_generate)} new topics...\n")

    for topic in topics_to_generate:
        print(f"⏳ Processing: {topic}")
        result = generate_topic_data(topic)
        
        if result:
            existing_data.append(result)
            # Save incrementally after every successful generation
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(existing_data, f, indent=2)
            print(f"✅ Saved: {topic}")
        else:
            print(f"⚠️  Skipped: {topic}")
            
        print("-" * 50)

    print(f"🎉 Generation complete! Total entries in {OUTPUT_FILE}: {len(existing_data)}")

if __name__ == "__main__":
    main()