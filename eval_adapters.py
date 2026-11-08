"""Score one or more Pythia-1B LoRA adapters using vLLM (on desktop GPU)
with an automatic fallback to Hugging Face (on laptop/CPU).

Usage:
  python eval_adapters_vllm.py ./adapter_answer:24 ./adapter_reasoning ./adapter_nonreasoning/ --save results.json
"""
import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "EleutherAI/pythia-1b-deduped"
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
  num = t.replace("$", "").replace(",", "").strip()
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


def parse_adapter_arg(arg):
  path, sep, tail = arg.rpartition(":")
  if sep and tail.isdigit():
    return path.rstrip("/"), int(tail)
  return arg.rstrip("/"), 1500


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", default=None, help="Held-out JSON file.")
  ap.add_argument("--hf", default=None, help="Datasets to use, e.g. arc,gsm8k or arc:200,gsm8k:50")
  ap.add_argument("--n", type=int, default=100, help="Questions per dataset (default 100)")
  ap.add_argument("--cache-dir", default=None, help="Hugging Face cache directory.")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("adapters", nargs="*", help="path or path:max_new_tokens")
  ap.add_argument("--batch-size", type=int, default=8, help="Batch size (used for HF fallback mode)")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--save", default=None)
  args = ap.parse_args()

  adapter_args = args.adapters
  if not adapter_args:
    raise SystemExit("No adapters given. Example: python eval_adapters_vllm.py ./adapter_cot ./adapter_answer:24")

  specs = []
  for a in adapter_args:
    path, max_new = parse_adapter_arg(a)
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
      raise SystemExit(f"'{path}' is not a valid PEFT adapter folder (missing adapter_config.json).")
    specs.append((path, max_new))

  items = []
  data_files = [args.data] if args.data else []
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

  use_vllm = torch.cuda.is_available()
  all_results = {}
  t0 = time.time()

  if use_vllm:
    print(f"\n[GPU Desktop Mode] Initializing vLLM engine with base model: {BASE_MODEL_ID}...")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
        max_model_len=CONTEXT,
        tensor_parallel_size=1,
    )

    for idx, (path, max_new) in enumerate(specs):
      print(f"\n=== Evaluating Adapter via vLLM: {path} (max_tokens={max_new}) ===")
      sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new, stop=["<|endoftext|>"])
      lora_request = LoRARequest(f"adapter_{idx}", idx + 1, path)

      outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

      results = []
      for j, output in enumerate(outputs):
        text = output.outputs[0].text
        n_tok = len(output.outputs[0].token_ids)
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
      all_results[path] = results

  else:
    print(f"\n[Laptop CPU Mode] No GPU found. Falling back to Hugging Face execution for local testing...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(BASE_Model_ID if 'BASE_Model_ID' in globals() else BASE_MODEL_ID, torch_dtype=torch.float32)
    model = None
    for i, (path, _) in enumerate(specs):
      name = f"adapter_{i}"
      if model is None:
        model = PeftModel.from_pretrained(base, path, adapter_name=name)
      else:
        model.load_adapter(path, adapter_name=name)
    model.eval()

    def generate_hf_batch(prompts_batch, max_new_tokens):
      enc = tokenizer(prompts_batch, return_tensors="pt", padding=True)
      prompt_len = enc["input_ids"].shape[1]
      budget = min(max_new_tokens, CONTEXT - prompt_len)
      if budget < 1:
        return [""] * len(prompts_batch), [0] * len(prompts_batch)
      with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=budget,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
      gen = out[:, prompt_len:]
      texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
      counts = (gen != tokenizer.pad_token_id).sum(dim=1).tolist()
      return texts, counts

    for i, (path, max_new) in enumerate(specs):
      adapter_name = f"adapter_{i}"
      model.set_adapter(adapter_name)
      print(f"\n=== Evaluating Adapter via Hugging Face: {path} (max_new_tokens={max_new}) ===")

      order = sorted(range(len(items)), key=lambda j: len(prompts[j]))
      results_temp = [None] * len(items)

      for start in range(0, len(order), args.batch_size):
        idx = order[start : start + args.batch_size]
        texts, counts = generate_hf_batch([prompts[j] for j in idx], max_new)
        for j, text, n_tok in zip(idx, texts, counts):
          pred = extract_answer(text)
          results_temp[j] = {
              "index": j,
              "domain": items[j].get("domain", "all"),
              "gold": items[j]["ground_truth"],
              "pred": pred,
              "correct": answers_match(pred, items[j]["ground_truth"]),
              "generated_tokens": n_tok,
              "generation": text,
          }
        print(f"  Processed {min(start + args.batch_size, len(order))}/{len(order)}", end="\r")
      print()
      all_results[path] = results_temp

  print(f"\nEvaluation completed in {time.time() - t0:.0f}s")

  summary = []
  for path, _ in specs:
    results = all_results[path]
    n = len(results)
    correct = sum(r["correct"] for r in results)
    no_format = sum(r["pred"] is None for r in results)
    avg_tok = sum(r["generated_tokens"] for r in results) / n
    summary.append((path, n, correct, no_format, avg_tok))

  print("\n" + "=" * 78)
  print(f"{'ADAPTER':<34} {'ACC':>6} {'CORRECT':>9} {'NO \"FINAL ANSWER\"':>18} {'AVG TOK':>8}")
  for path, n, c, nf, avg in summary:
    print(f"{path:<34} {c / n:>6.0%} {f'{c}/{n}':>9} {nf:>18} {avg:>8.0f}")
  print(f"{'(most-common-answer baseline)':<34} {majority:.0%}")

  if args.save:
    with open(args.save, "w", encoding="utf-8") as f:
      json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved generations to {args.save}")


if __name__ == "__main__":
  main()
