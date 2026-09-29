import json
from collections import Counter
import os

# Get all JSON files in the current directory
json_files = [f for f in os.listdir() if f.endswith('.json')]

for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        
        print(f"\nAnalyzing {file}:")
        print(f"Total questions: {len(data)}")
        print(f"\nBy topic:")
        topics = Counter(q.get("topic") for q in data)
        for topic, count in topics.most_common():
            print(f"  {topic}: {count}")

        print(f"\nBy difficulty:")
        difficulties = Counter(q.get("difficulty") for q in data)
        for diff, count in difficulties.most_common():
            print(f"  {diff}: {count}")

        print(f"\nBy question type:")
        types = Counter(q.get("question_type") for q in data)
        for qtype, count in types.most_common():
            print(f"  {qtype}: {count}")