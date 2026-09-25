import json
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

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

# 3. LOAD AND FORMAT DATASET (Bypasses the buggy Hugging Face JSON loader)
print("📂 Loading and formatting dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

formatted_data = []
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
    formatted_data.append({"text": text})

# Create dataset directly from list, dodging the dill/pickle hashing bug
dataset = Dataset.from_list(formatted_data)
print(f"✅ Successfully loaded {len(dataset)} examples.")

# 4. TRAIN
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    dataset_num_proc=2,
    packing=False,
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
    ),
)

print("🎵 Starting training...")
trainer.train()

# 5. EXPORT TO GGUF (Ready for Ollama)
print("💾 Exporting to GGUF (Q4_K_M quantization)...")
model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")
print(f"✅ Done! Model saved to {OUTPUT_DIR}")