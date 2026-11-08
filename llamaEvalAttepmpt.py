"""Automated LoRA adapter evaluation script using in-memory model merging
and llama-cpp-python GGUF evaluation.

Usage:
  python eval_adapters_merged.py --base-model EleutherAI/pythia-1b-deduped ./adapter_cot ./adapter_answer:24
"""
import argparse
import os
import subprocess
import sys
import json
import re
import time
from collections import Counter, defaultdict

from llama_cpp import Llama

CONTEXT = 2048

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Solve the following problem. "
    "End with 'FINAL ANSWER:' followed by only the answer.\n\n"
    "{problem}\n\n"
    "### Response:\n"
)

FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)


def normalize(text):
  t = text.strip().rstrip(".").strip()
  m = re.fullmatch(r"\(?([A-Za-z])\)?", t)
  if m:
    return m.group(1).upper()
  num = t.replace("$", "",).replace(",", "").strip()
  try:
    f = float(num)
    return str(int(f)) if f == int(f) else str(f)
  except ValueError:
    return t.lower()


def extract_answer(generation):
  matches = FINAL_RE.findall(generation)
  return matches[-1].strip() if matches else None


def answers_match(pred, gold):
  return pred is not None and normalize(pred) == normalize(gold)


# ---------------------------------------------------------------------------
# Automatic Conversion Helpers via Merging
# ---------------------------------------------------------------------------
def ensure_llama_cpp_repo():
  """Clones llama.cpp repository locally if conversion scripts are missing."""
  if not os.path.isdir("llama.cpp"):
    print("Cloning llama.cpp repository for conversion scripts...")
    subprocess.run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "llama.cpp/requirements.txt"], check=True)


def merge_and_convert_adapter(base_model_id, adapter_arg_path):
  """Merges a PEFT LoRA adapter into the base model using PyTorch,
     saves it as a temporary HF folder, and converts it to a standalone GGUF model.
  """
  path, sep, tail = adapter_arg_path.rpartition(":")
  max_new = int(tail) if sep and tail.isdigit() else 1500
  actual_path = path if sep and tail.isdigit() else adapter_arg_path

  # If it's already a .gguf file, return it directly
  if actual_path.endswith(".gguf") and os.path.isfile(actual_path):
    return actual_path, max_new

  safe_adapter_name = actual_path.replace("/", "_").replace(".", "_")
  gguf_filename = f"model_{safe_adapter_name}.gguf"

  if os.path.isfile(gguf_filename):
    print(f"Found existing converted GGUF for adapter: {gguf_filename}")
    return gguf_filename, max_new

  print(f"\n[1/3] Loading base model '{base_model_id}' to merge with '{actual_path}'...")
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from peft import PeftModel

  tokenizer = AutoTokenizer.from_pretrained(base_model_id)
  base_model = AutoModelForCausalLM.from_pretrained(
      base_model_id,
      torch_dtype=torch.float16,
      device_map="auto"
  )

  print(f"[2/3] Merging LoRA weights from {actual_path}...")
  model = PeftModel.from_pretrained(base_model, actual_path)
  merged_model = model.merge_and_unload()

  temp_save_dir = f"./temp_merged_{safe_adapter_name}"
  merged_model.save_pretrained(temp_save_dir)
  tokenizer.save_pretrained(temp_save_dir)

  # Clear PyTorch memory to free up VRAM for llama.cpp later
  del base_model, model, merged_model
  torch.cuda.empty_cache()

  print(f"[3/3] Converting merged model to GGUF ({gguf_filename})...")
  ensure_llama_cpp_repo()
  convert_script = os.path.join("llama.cpp", "convert_hf_to_gguf.py")

  subprocess.run([
      sys.executable, convert_script,
      temp_save_dir,
      "--outfile", gguf_filename,
      "--outtype", "f16"
  ], check=True)

  return gguf_filename, max_new


# ---------------------------------------------------------------------------
# Hugging Face Questions Loader
# ---------------------------------------------------------------------------
LETTERS = "ABCDEFGHIJ"

def format_mcq(question, options):
  body = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
  return f"{question}\n\n{body}\n\nChoose the single best option."

def load_arc(n, seed, cache_dir=None):
  from datasets import load_dataset
  ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=cache_dir).shuffle(seed=seed)
  out = []
  for ex in ds:
    choices = ex.get("choices") or {}
    labels, texts = list(choices.get("label", [])), list(choices.get("text", []))
    key = ex.get("answerKey")
    if not texts or key not in labels:
      continue
    out.append({"problem": format_mcq(ex["question"], texts),
                "ground_truth": LETTERS[labels.index(key)],
                "domain": "arc"})
    if len(out) >= n:
      break
  return out

def load_gsm8k(n, seed, cache_dir=None):
  from datasets import load_dataset
  ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir).shuffle(seed=seed)
  out = []
  for ex in ds:
    m = re.search(r"####\s*(.+)", ex.get("answer", ""))
    if not m or not ex.get("question"):
      continue
    out.append({"problem": ex["question"],
                "ground_truth": m.group(1).strip().replace(",", ""),
                "domain": "gsm8k"})
    if len(out) >= n:
      break
  return out

HF_LOADERS = {"arc": load_arc, "gsm8k": load_gsm8k}

def parse_hf_spec(spec, default_count=100):
  pairs = []
  for part in spec.split(","):
    part = part.strip()
    if not part:
      continue
    name, _, count = part.partition(":")
    if name not in HF_LOADERS:
      raise SystemExit(f"Unknown --hf dataset '{name}'. Available: {', '.join(HF_LOADERS)}")
    pairs.append((name, int(count) if count else default_count))
  return pairs


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--base-model", default="EleutherAI/pythia-1b-deduped",
                  help="Hugging Face repo ID or path to base model.")
  ap.add_argument("--data", default=None, help="Held-out JSON file.")
  ap.add_argument("--hf", default=None, help="Datasets to use (e.g. arc,gsm8k or arc:100)")
  ap.add_argument("--n", type=int, default=100, help="Questions per dataset (default 100)")
  ap.add_argument("--cache-dir", default=None, help="Hugging Face cache directory.")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("adapters", nargs="*", help="Adapter folder path or path:max_new_tokens")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--save", default=None)
  args = ap.parse_args()

  if not args.adapters:
    raise SystemExit("No adapters given. Example: python eval_adapters_merged.py ./adapter_cot ./adapter_answer:24")

  # 1. Load evaluation datasets first
  data_files = [args.data] if args.data else []
  items = []
  for data_file in data_files:
    with open(data_file, "r", encoding="utf-8") as f:
      items.extend(json.load(f))
    print(f"Loaded {len(items)} questions from {data_file}")

  hf_spec = args.hf or (None if data_files else "arc:100,gsm8k:100")
  if hf_spec:
    for name, count in parse_hf_spec(hf_spec, args.n):
      got = HF_LOADERS[name](count, args.seed, args.cache_dir)
      print(f"Loaded {len(got)} {name} test questions from Hugging Face")
      items.extend(got)

  if not items:
    raise SystemExit("No questions to evaluate.")
  if args.limit:
    items = items[: args.limit]

  prompts = [PROMPT_TEMPLATE.format(problem=it["problem"].strip()) for it in items]

  gold_counts = Counter(normalize(it["ground_truth"]) for it in items)
  majority = gold_counts.most_common(1)[0][1] / len(items)
  print(f"{len(items)} held-out questions. Baseline (most common answer): {majority:.0%}")

  all_results = {}
  t0 = time.time()

  # 2. Loop through each adapter: Merge -> Convert to GGUF -> Evaluate via llama-cpp-python
  for original_arg in args.adapters:
    gguf_model_path, max_new = merge_and_convert_adapter(args.base_model, original_arg)

    print(f"\n=== Evaluating GGUF Model for: {original_arg} (max_tokens={max_new}) ===")

    llm = Llama(
        model_path=gguf_model_path,
        n_ctx=CONTEXT,
        n_gpu_layers=-1, # Offload to GPU automatically if available
        verbose=False
    )

    results = []
    for j, prompt in enumerate(prompts):
      output = llm(
          prompt,
          max_tokens=max_new,
          temperature=0.0,
          stop=["<|endoftext|>"]
      )
      text = output["choices"][0]["text"]
      n_tok = output["usage"]["completion_tokens"]

      pred = extract_answer(text)
      results.append({
          "index": j,
          "domain": items[j].get("domain", "all"),
          "gold": items[j]["ground_truth"],
          "pred": pred,
          "correct": answers_match(pred, items[j]["ground_truth"]),
          "generated_tokens": n_tok,
          "generation": text,
      })
      print(f"  Processed {j + 1}/{len(prompts)}", end="\r")
    print()

    all_results[original_arg] = results
    # Free up memory before processing the next adapter
    del llm

  print(f"\nEvaluation completed in {time.time() - t0:.0f}s")

  # 3. Print Summary
  summary = []
  for original_arg in args.adapters:
    results = all_results[original_arg]
    n = len(results)
    correct = sum(r["correct"] for r in results)
    no_format = sum(r["pred"] is None for r in results)
    avg_tok = sum(r["generated_tokens"] for r in results) / n
    summary.append((original_arg, n, correct, no_format, avg_tok))

  print("\n" + "=" * 78)
  print(f"{'ADAPTER':<34} {'ACC':>6} {'CORRECT':>9} {'NO "FINAL ANSWER"':>18} {'AVG TOK':>8}")
  for path, n, c, nf, avg in summary:
    print(f"{path:<34} {c / n:>6.0%} {f'{c}/{n}':>9} {nf:>18} {avg:>8.0f}")
  print(f"{'(most-common-answer baseline)':<34} {majority:.0%}")

  if args.save:
    with open(args.save, "w", encoding="utf-8") as f:
      json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved generations to {args.save}")


if __name__ == "__main__":
  main()
