import json
import os
import shutil
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Try importing Unsloth, fallback to standard HF if on Mac and Unsloth fails
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    USE_UNSLOTH = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig
    USE_UNSLOTH = False
    def is_bfloat16_supported():
        return torch.backends.mps.is_available() # Fallback check

# Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" # Changed: bnb-4bit does not work on Mac
DATASET_PATH = "final_master_dataset.json"
OUTPUT_DIR = "./python_tutor_adapter"

print("📥 Loading Qwen2.5-3B in 16-bit precision (Mac Compatible)...")
if USE_UNSLOTH:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        dtype=torch.float16, # Changed: Mac uses float16 reliably
        load_in_4bit=False,  # Changed: bitsandbytes 4-bit is CUDA only
    )
    model = FastLanguageModel.get_peft_model(
        model, r=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    peft_config = LoraConfig(r=32, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

# ... [Keep the format_elaborate_response and formatting_prompts_func exactly as you had them] ...

# Load and format dataset
print("📂 Loading and formatting dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

texts = []
for item in raw_data:
    instruction = item.get("instruction", "")
    output = item.get("output", {})
    formatted = formatting_prompts_func({"instruction": [instruction], "output": [output]})
    texts.extend(formatted["text"])

print(f"✅ Loaded {len(texts)} examples. Tokenizing...")
encodings = tokenizer(texts, truncation=True, padding=True, max_length=2048, return_tensors="pt")

class SimpleDataset(Dataset):
    def __init__(self, encodings): # Fixed: Added double underscores
        self.encodings = encodings
    def __len__(self):            # Fixed: Added double underscores
        return len(self.encodings.input_ids)
    def __getitem__(self, idx):   # Fixed: Added double underscores
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = SimpleDataset(encodings)

# Check if MPS (Mac) is being used to adjust training args
is_mps = torch.backends.mps.is_available()

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=TrainingArguments(
        per_device_train_batch_size=1, # Reduced batch size for Mac RAM safety
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported() and not is_mps, 
        bf16=is_bfloat16_supported() and not is_mps,
        logging_steps=10,
        optim="adamw_torch", # Changed: adamw_8bit relies on bitsandbytes (CUDA only)
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
    ),
)

print("🎵 Starting training...")
trainer.train()

print("💾 Saving LoRA adapter...")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA adapter saved to {OUTPUT_DIR}")