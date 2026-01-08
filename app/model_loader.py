import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL = "google/flan-t5-base"
ADAPTER_PATH = "medical_coding_assistant_v3"


DEVICE = "cpu"

# Prevent CPU overuse
torch.set_num_threads(2)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading CLEAN base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
base_model.eval()

# loading second time to ensure proper setup for PEFT
print("Loading LoRA adapter...")
print("Loading ICD model (base + LoRA)...")
icd_base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(icd_base_model, ADAPTER_PATH)
model.eval()


print("Model loaded successfully.")
