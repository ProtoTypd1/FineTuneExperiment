import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "EleutherAI/pythia-1b-deduped"
ADAPTER_PATH = "./pythia_1b_lora_adapter"

# Must match train_lora.py exactly.
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Solve the following logic problem and provide the final option letter.\n\n"
    "{problem}\n\n"
    "### Response:\n"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = "<|padding|>"  # same dedicated pad token as training

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

sample_problem = """You have a list of tasks. Task A must be completed before Task B. Task B must be completed before Task C. Which task should be completed first?
Options:
(A) Task B
(B) Task A
(C) Task C"""

formatted_prompt = PROMPT_TEMPLATE.format(problem=sample_problem)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
  outputs = model.generate(
      **inputs,
      max_new_tokens=20,
      do_sample=False,  # greedy; temperature is ignored here so it's removed
      eos_token_id=tokenizer.eos_token_id,
      pad_token_id=tokenizer.pad_token_id,
  )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
).strip()

print("--- Model Response ---")
print(response)

match = re.search(r"FINAL ANSWER:\s*\(?([A-Za-z])\)?", response)
print("Parsed answer:", match.group(1).upper() if match else None)
