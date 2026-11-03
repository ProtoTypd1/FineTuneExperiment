import json
import re

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ==========================================
# 0. Config
# ==========================================
MODEL_ID = "EleutherAI/pythia-1b-deduped"
DATASET_PATH = "your_dataset.json"  # list of {"problem": ..., "ground_truth": ...}
OUTPUT_DIR = "./pythia_1b_lora_adapter"

use_bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

# ==========================================
# 1. Tokenizer (set up BEFORE building the dataset, since we need eos_token)
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Use a dedicated pad token. If pad == eos, collators mask the EOS from the loss
# and the model never learns to stop.
pad_id = tokenizer.convert_tokens_to_ids("<|padding|>")
if pad_id is None or pad_id == tokenizer.eos_token_id:
  raise RuntimeError("'<|padding|>' not found in tokenizer vocab; check your tokenizer.")
tokenizer.pad_token = "<|padding|>"

# ==========================================
# 2. Dataset loading & formatting (prompt/completion for completion-only loss)
# ==========================================
# NOTE: this template must match inference.py exactly.
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Solve the following logic problem and provide the final option letter.\n\n"
    "{problem}\n\n"
    "### Response:\n"
)


def normalize_answer(ground_truth):
  """Reduce '(B) Task A' / 'B' / 'b)' to just the letter 'B'."""
  s = str(ground_truth).strip()
  m = re.match(r"^\(?([A-Za-z])\)?(?:\W|$)", s)
  return m.group(1).upper() if m else s


def load_and_format_dataset(json_file_path):
  with open(json_file_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

  if isinstance(raw_data, dict):
    raw_data = [raw_data]

  samples = []
  odd_answers = 0
  for item in raw_data:
    problem = item.get("problem", "")
    answer = normalize_answer(item.get("ground_truth", ""))
    if not re.fullmatch(r"[A-Z]", answer):
      odd_answers += 1

    samples.append({
        "prompt": PROMPT_TEMPLATE.format(problem=problem),
        "completion": f"FINAL ANSWER: {answer}{tokenizer.eos_token}",
    })

  if odd_answers:
    print(f"WARNING: {odd_answers}/{len(samples)} answers are not a single letter.")
  return Dataset.from_list(samples)


dataset = load_and_format_dataset(DATASET_PATH)

# Hold out 10% to watch for overfitting (needs more than a handful of examples)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = split["train"], split["test"]
print(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

# ==========================================
# 3. Model (native bf16/fp16, no quantization needed for 1B)
# ==========================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
)

# ==========================================
# 4. LoRA config (GPT-NeoX module names) -- passed to the trainer only,
#    so the model is wrapped exactly once.
# ==========================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ==========================================
# 5. Training arguments
# ==========================================
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=1024,  # was max_seq_length in older TRL
    completion_only_loss=True,  # loss on the completion only, not the prompt
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch size 16
    learning_rate=3e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    logging_steps=10,
    eval_strategy="epoch",  # was evaluation_strategy in older transformers
    save_strategy="epoch",
    bf16=use_bf16,
    fp16=not use_bf16,
    optim="adamw_torch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,  # was tokenizer= in older TRL
)

trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete! Adapter saved to {OUTPUT_DIR}")
