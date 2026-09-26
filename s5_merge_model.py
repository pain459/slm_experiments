from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil
import os

# Configuration
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # Full 16-bit base model
ADAPTER_DIR = "./lora_adapter"
OUTPUT_DIR = "./merged_hf_model"

print("📥 Loading base model in 16-bit precision...")
# Load the FULL precision base model (not 4-bit)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype="auto",
    device_map="cpu"  # Merge on CPU to avoid VRAM issues
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("📥 Loading LoRA adapter...")
# Load the adapter on top of the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("🔀 Merging LoRA weights into base model...")
# Merge the adapter into the base model (in memory)
merged_model = model.merge_and_unload()

print("💾 Saving merged model...")
# Clean output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Save the fully merged 16-bit model
merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Merged model saved to {OUTPUT_DIR}")
print("🎉 Ready for GGUF conversion!")