import json
import torch
from torch.utils.data import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Configuration
MODEL_ID = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
DATASET_PATH = "final_master_dataset.json"
OUTPUT_DIR = "./python_tutor_adapter"

# Load model
print("📥 Loading Qwen2.5-3B in 4-bit precision...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,  # Increased for elaborate responses
    dtype=None,
    load_in_4bit=True,
)

# Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Higher rank for larger dataset
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Format structured JSON to elaborate markdown
def format_elaborate_response(output_dict):
    """Converts structured output to comprehensive markdown response, handling messy LLM data."""
    parts = []
    
    # Opening
    parts.append("Let me explain this comprehensively.\n")
    
    # Definition (only if it exists and is not empty)
    if output_dict.get("definition") and str(output_dict["definition"]).strip():
        parts.append(f"## Definition\n{output_dict['definition']}\n")
    
    # Detailed explanation (only if it exists and is not empty)
    if output_dict.get("explanation") and str(output_dict["explanation"]).strip():
        parts.append(f"## Detailed Explanation\n{output_dict['explanation']}\n")
    
    # Examples with code (defensive against strings, empty dicts, or missing keys)
    if output_dict.get("examples") and isinstance(output_dict["examples"], list):
        valid_examples = []
        for ex in output_dict["examples"]:
            if isinstance(ex, dict):
                if ex.get("code") or ex.get("description"):
                    valid_examples.append(ex)
            elif isinstance(ex, str) and ex.strip():
                # Fallback if the LLM output a raw string instead of a dict
                valid_examples.append({"code": ex, "description": "Example"})
        
        if valid_examples:
            parts.append("## Practical Examples\n")
            for i, ex in enumerate(valid_examples, 1):
                desc = ex.get("description", "Code demonstration") if isinstance(ex, dict) else "Example"
                code = ex.get("code", "") if isinstance(ex, dict) else str(ex)
                
                parts.append(f"**Example {i}: {desc}**")
                if code.strip():
                    parts.append(f"```python\n{code}\n```\n")
                else:
                    parts.append("*(Conceptual example, no code required)*\n")
    
    # Common mistakes (filter out empty strings or non-lists)
    if output_dict.get("common_mistakes") and isinstance(output_dict["common_mistakes"], list):
        mistakes = [m for m in output_dict["common_mistakes"] if isinstance(m, str) and m.strip()]
        if mistakes:
            parts.append("## Common Pitfalls to Avoid\n")
            for mistake in mistakes:
                parts.append(f"- {mistake}")
            parts.append("")
    
    # Best practices (filter out empty strings or non-lists)
    if output_dict.get("best_practices") and isinstance(output_dict["best_practices"], list):
        practices = [p for p in output_dict["best_practices"] if isinstance(p, str) and p.strip()]
        if practices:
            parts.append("## Best Practices\n")
            for practice in practices:
                parts.append(f"- {practice}")
            parts.append("")
    
    # Closing
    parts.append("This should give you a solid understanding. Would you like me to elaborate on any specific aspect or show you more advanced examples?")
    
    return "\n".join(parts)

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    
    for instruction, output in zip(instructions, outputs):
        # Convert structured output to elaborate markdown
        if isinstance(output, dict):
            formatted_output = format_elaborate_response(output)
        else:
            formatted_output = str(output)
        
        # ChatML format with comprehensive system prompt
        text = (
            "<|im_start|>system\n"
            "You are an expert Python programmer and patient teacher with deep knowledge of Python (from basics to expert level) and Data Structures & Algorithms. "
            "When explaining concepts, provide comprehensive, well-structured responses that include:\n"
            "- A clear definition of the core concept\n"
            "- A detailed explanation of how it works and why it matters\n"
            "- Step-by-step breakdowns with complete, runnable code examples\n"
            "- Common pitfalls and mistakes to avoid\n"
            "- Professional best practices\n"
            "Use markdown formatting (headers, code blocks, lists) to organize your response clearly. "
            "Be conversational, educational, and engaging. Always end with an offer to elaborate further.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{instruction}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{formatted_output}\n"
            "<|im_end|>"
        )
        texts.append(text)
    
    return {"text": texts}

# Load and format dataset
print("📂 Loading and formatting dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

texts = []
for item in raw_data:
    instruction = item.get("instruction", "")
    output = item.get("output", {})
    
    formatted = formatting_prompts_func({
        "instruction": [instruction],
        "output": [output]
    })
    texts.extend(formatted["text"])

print(f"✅ Loaded {len(texts)} examples. Tokenizing...")

# Tokenize
encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=2048,
    return_tensors="pt"
)

class SimpleDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings.input_ids)
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = SimpleDataset(encodings)
print(f"✅ Dataset ready with {len(dataset)} tokenized examples.")

# Training configuration (adjusted for larger dataset)
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,  # 3 epochs for comprehensive learning
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,  # Log every 10 steps
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Better convergence
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
    ),
)

print("🎵 Starting training...")
trainer.train()

# Save LoRA adapter only
print("💾 Saving LoRA adapter...")
import shutil
import os
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ LoRA adapter saved to {OUTPUT_DIR}")
print("🎉 Training complete! Now run the merge script.")