import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "EleutherAI/pythia-1b-deduped"
ADAPTER_PATH = sys.argv[1] if len(sys.argv) > 1 else "./pythia_1b_lora_adapter"
# Reasoning adapters need room to write; answer-only adapters stop after ~10 tokens.
MAX_NEW_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

# Must match train_lora.py exactly.
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Solve the following problem. "
    "End with 'FINAL ANSWER:' followed by only the answer.\n\n"
    "{problem}\n\n"
    "### Response:\n"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = "<|padding|>"  # same dedicated pad token as training

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    attn_implementation="sdpa",
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

sample_problem = """George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?

A) dry palms
B) wet palms
C) palms covered with oil
D) palms covered with lotion

Choose the single best option."""

formatted_prompt = PROMPT_TEMPLATE.format(problem=sample_problem)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
  outputs = model.generate(
      **inputs,
      max_new_tokens=MAX_NEW_TOKENS,
      do_sample=False,  # greedy; temperature is ignored here so it's removed
      eos_token_id=tokenizer.eos_token_id,
      pad_token_id=tokenizer.pad_token_id,
  )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
).strip()

print("--- Model Response ---")
print(response)

matches = re.findall(r"FINAL ANSWER:\s*(.+)", response)
print("Parsed answer:", matches[-1].strip() if matches else None)
