import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ==========================================
# 0. Config
# ==========================================
MODEL_ID = "EleutherAI/pythia-1b-deduped"
import sys
# Pass the dataset and an output dir per arm, e.g.:
#   python train_lora.py data_cot.json ./adapter_cot
#   python train_lora.py data_answer.json ./adapter_answer
DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "data_cot.json"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./pythia_1b_lora_adapter"

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
    "Solve the following problem. "
    "End with 'FINAL ANSWER:' followed by only the answer.\n\n"
    "{problem}\n\n"
    "### Response:\n"
)


def load_and_format_dataset(json_file_path):
  with open(json_file_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

  if isinstance(raw_data, dict):
    raw_data = [raw_data]

  samples = []
  for item in raw_data:
    problem = item.get("problem", "")
    answer = str(item.get("ground_truth", "")).strip()
    # CoT arm: item has a "response" (written reasoning ending in "FINAL ANSWER: X").
    # Answer-only arm: no "response", so the target is just "FINAL ANSWER: X".
    response = (item.get("response") or "").strip()
    target = response if response else f"FINAL ANSWER: {answer}"

    samples.append({
        "prompt": PROMPT_TEMPLATE.format(problem=problem.strip()),
        "completion": f"{target}{tokenizer.eos_token}",
    })

  return Dataset.from_list(samples)


MAX_LENGTH = 2048  # Pythia's context window

dataset = load_and_format_dataset(DATASET_PATH)

# Truncation would cut off the FINAL ANSWER line, so drop samples that don't fit.
def _fits(ex):
  return len(tokenizer(ex["prompt"] + ex["completion"])["input_ids"]) <= MAX_LENGTH

n_before = len(dataset)
dataset = dataset.filter(_fits)
print(f"Dropped {n_before - len(dataset)} samples longer than {MAX_LENGTH} tokens; {len(dataset)} remain.")

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
    attn_implementation="sdpa",  # PyTorch fused attention; no flash-attn install needed
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
# Explicit warmup steps (3% of training). Works across transformers versions:
# newer ones removed warmup_ratio, older ones don't accept a fractional warmup_steps.
BATCH_SIZE, GRAD_ACCUM, EPOCHS = 4, 4, 3
steps_per_epoch = max(1, -(-len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)))
WARMUP_STEPS = max(1, int(0.03 * steps_per_epoch * EPOCHS))

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=MAX_LENGTH,
    completion_only_loss=True,  # loss on the completion only, not the prompt
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=GRAD_ACCUM,  # effective batch size 16
    learning_rate=3e-4,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    num_train_epochs=EPOCHS,
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
