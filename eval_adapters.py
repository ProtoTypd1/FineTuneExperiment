"""Score one or more Pythia-1B LoRA adapters on held-out questions.

Usage:
  python eval_adapters.py data_heldout.json ./adapter_answer:24 ./adapter_cot ./adapter_think

Each adapter is  path  or  path:max_new_tokens  (default 1500). Use a small budget
(~24) for answer-only adapters and ~1500 for reasoning adapters.

Options:
  --batch-size N   prompts generated at once (default 8; lower it if you run out of memory)
  --limit N        only evaluate the first N questions
  --save FILE      write every generation and its score to FILE (JSON)
"""
import argparse
import json
import re
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "EleutherAI/pythia-1b-deduped"
CONTEXT = 2048

# Must match train_lora.py / inference.py exactly.
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Solve the following problem. "
    "End with 'FINAL ANSWER:' followed by only the answer.\n\n"
    "{problem}\n\n"
    "### Response:\n"
)

FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
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
# Generation
# ---------------------------------------------------------------------------
def generate_batch(model, tokenizer, prompts, max_new_tokens):
  enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
  prompt_len = enc["input_ids"].shape[1]
  budget = min(max_new_tokens, CONTEXT - prompt_len)
  if budget < 1:  # prompt alone fills the context window
    return [""] * len(prompts), [0] * len(prompts)

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


def parse_adapter_arg(arg):
  path, sep, tail = arg.rpartition(":")
  if sep and tail.isdigit():
    return path, int(tail)
  return arg, 1500


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("data", help="Held-out JSON from prepare_data.py (problem, ground_truth).")
  ap.add_argument("adapters", nargs="+", help="path or path:max_new_tokens")
  ap.add_argument("--batch-size", type=int, default=8)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--save", default=None)
  args = ap.parse_args()

  with open(args.data, "r", encoding="utf-8") as f:
    items = json.load(f)
  if args.limit:
    items = items[: args.limit]
  prompts = [PROMPT_TEMPLATE.format(problem=it["problem"].strip()) for it in items]

  gold_counts = Counter(normalize(it["ground_truth"]) for it in items)
  majority = gold_counts.most_common(1)[0][1] / len(items)
  print(f"{len(items)} held-out questions. Always guessing the most common answer: {majority:.0%}")

  tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
  tokenizer.pad_token = "<|padding|>"
  tokenizer.padding_side = "left"  # required for batched generation with a decoder-only model

  base = AutoModelForCausalLM.from_pretrained(
      BASE_MODEL_ID,
      torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
      attn_implementation="sdpa",
      device_map="auto",
  )

  specs = [parse_adapter_arg(a) for a in args.adapters]
  model = None
  for i, (path, _) in enumerate(specs):
    name = f"adapter_{i}"
    if model is None:
      model = PeftModel.from_pretrained(base, path, adapter_name=name)
    else:
      model.load_adapter(path, adapter_name=name)
  model.eval()

  summary, all_results = [], {}
  for i, (path, max_new) in enumerate(specs):
    model.set_adapter(f"adapter_{i}")
    print(f"\n=== {path}  (max_new_tokens={max_new}) ===")

    results = []
    # Sort by prompt length so each batch has similar lengths (less padding).
    order = sorted(range(len(items)), key=lambda j: len(prompts[j]))
    for start in range(0, len(order), args.batch_size):
      idx = order[start : start + args.batch_size]
      texts, counts = generate_batch(model, tokenizer, [prompts[j] for j in idx], max_new)
      for j, text, n_tok in zip(idx, texts, counts):
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
      print(f"  {min(start + args.batch_size, len(order))}/{len(order)}", end="\r")
    print()

    results.sort(key=lambda r: r["index"])
    all_results[path] = results

    n = len(results)
    correct = sum(r["correct"] for r in results)
    no_format = sum(r["pred"] is None for r in results)
    avg_tok = sum(r["generated_tokens"] for r in results) / n
    summary.append((path, n, correct, no_format, avg_tok))

    by_domain = defaultdict(lambda: [0, 0])
    for r in results:
      by_domain[r["domain"]][0] += r["correct"]
      by_domain[r["domain"]][1] += 1
    for d, (c, t) in sorted(by_domain.items()):
      print(f"  {d:<12} {c}/{t} = {c / t:.0%}")

  print("\n" + "=" * 78)
  print(f"{'ADAPTER':<34} {'ACC':>6} {'CORRECT':>9} {'NO "FINAL ANSWER"':>18} {'AVG TOK':>8}")
  for path, n, c, nf, avg in summary:
    print(f"{path:<34} {c / n:>6.0%} {f'{c}/{n}':>9} {nf:>18} {avg:>8.0f}")
  print(f"{'(most-common-answer baseline)':<34} {majority:>6.0%}")

  if args.save:
    with open(args.save, "w", encoding="utf-8") as f:
      json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved generations to {args.save}")


if __name__ == "__main__":
  main()
