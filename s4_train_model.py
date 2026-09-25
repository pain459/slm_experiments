import json
import torch
from torch.utils.data import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1. CONFIGURATION
MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DATASET_PATH = "dataset.json"
OUTPUT_DIR = "./trained_model_gguf"

# 2. LOAD MODEL (4-bit quantized to save VRAM)
print("📥 Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=1024,
    dtype=None, # Auto-detects fp16 or bf16
    load_in_4bit=True,
)

# Attach LoRA adapters (the learnable part)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 3. LOAD AND FORMAT DATASET (Pure PyTorch, bypasses HF datasets/dill entirely)
print("📂 Loading and formatting dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

texts = []
for item in raw_data:
    instruction = item.get("instruction", "")
    output = item.get("output", "")
    
    # ChatML format for Qwen 2.5
    text = (
        "<|im_start|>system\n"
        "You are an expert tutor. Always provide the direct answer first, followed by a clear, step-by-step explanation of the reasoning.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{output}\n"
        "<|im_end|>"
    )
    texts.append(text)

print(f"✅ Loaded {len(texts)} examples. Tokenizing...")

# Tokenize all at once (much faster and avoids per-item overhead)
encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=1024,
    return_tensors="pt"
)

# Native PyTorch Dataset (Zero dill/pickle dependencies)
class SimpleDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings.input_ids)
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = SimpleDataset(encodings)
print(f"✅ Dataset ready with {len(dataset)} tokenized examples.")

# 4. TRAIN
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,               # 3 full passes over your data
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
    ),
)

print("🎵 Starting training...")
trainer.train()

# 5. SAVE LoRA ADAPTER ONLY (Fast, no merging, no quantization issues)
print("💾 Saving LoRA adapter...")
adapter_dir = "./lora_adapter"

import shutil
import os
if os.path.exists(adapter_dir):
    shutil.rmtree(adapter_dir)

# Save just the adapter weights (tiny file, ~50MB)
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print(f"✅ LoRA adapter saved to {adapter_dir}")
print("🎉 Training complete! Now run the merge script to create the final model.")