"""Score one or more Pythia-1B LoRA adapters on held-out questions.

Usage:
  # default: 100 random test-split questions from EACH dataset (arc, gsm8k)
  python eval_adapters.py ./adapter_answer:24 ./adapter_cot ./adapter_think

  # one number for every dataset
  python eval_adapters.py --n 50 ./adapter_cot

  # or per-dataset counts / a subset of datasets
  python eval_adapters.py --hf arc:200,gsm8k:50 ./adapter_cot
  python eval_adapters.py --hf gsm8k --n 300 ./adapter_cot

  # reuse datasets already downloaded by the generator (and never touch the network)
  HF_DATASETS_OFFLINE=1 python eval_adapters.py --cache-dir ./datasets ./adapter_cot

  # download + save the questions once, then run fully offline from the saved file
  python eval_adapters.py --export questions.json
  python eval_adapters.py --data questions.json ./adapter_cot

  # use your own held-out file instead (from prepare_data.py), alone or combined with --hf
  python eval_adapters.py --data data_heldout.json ./adapter_cot

Each adapter is  path  or  path:max_new_tokens  (default 1500). Use a small budget
(~24) for answer-only adapters and ~1500 for reasoning adapters.

By default all adapters are loaded at once and their jobs are mixed into the same batches
(PEFT "mixed adapter batches": the base-model math is shared, each row uses its own LoRA).
Use --sequential to run one adapter at a time instead.

Options:
  --n N            questions per dataset (default 100)
  --cache-dir DIR  Hugging Face cache location
  --hf SPEC        datasets, comma separated, each optionally with :count (available: arc, gsm8k).
                   Questions come from each dataset's TEST split, which the trajectory
                   generator never used (it read the train splits). Seeded, so every
                   run and every adapter sees the same questions.
  --data FILE      held-out JSON with problem / ground_truth (optional)
  --seed N         shuffle seed for --hf sampling (default 42)
  --sequential     one adapter at a time (fallback if mixed batching errors on your PEFT version)
  --batch-size N   sequences generated at once (default 24 mixed / 8 sequential; lower it
                   if you run out of memory)
  --limit N        only evaluate the first N questions
  --save FILE      write every generation and its score to FILE (JSON)
"""
import argparse
import json
import os
import re
import time
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
# Hugging Face questions (formatted exactly like generate_trajectories.py)
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
                "ground_truth": LETTERS[labels.index(key)],  # options are re-lettered A, B, C...
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
# Generation
# ---------------------------------------------------------------------------
def generate_batch(model, tokenizer, prompts, max_new_tokens, adapter_names=None):
  enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
  prompt_len = enc["input_ids"].shape[1]
  budget = min(max_new_tokens, CONTEXT - prompt_len)
  if budget < 1:  # prompt alone fills the context window
    return [""] * len(prompts), [0] * len(prompts)

  extra = {"adapter_names": adapter_names} if adapter_names is not None else {}
  with torch.no_grad():
    out = model.generate(
        **enc,
        **extra,
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
  ap.add_argument("--data", default=None,
                  help="Held-out JSON from prepare_data.py (problem, ground_truth).")
  ap.add_argument("--hf", default=None,
                  help="datasets to use, e.g. arc,gsm8k or arc:200,gsm8k:50 "
                       "(default when neither --data nor --hf is given: all of them)")
  ap.add_argument("--n", type=int, default=100,
                  help="questions per dataset, for every --hf entry without its own :count (default 100)")
  ap.add_argument("--cache-dir", default=None,
                  help="where Hugging Face datasets are cached, e.g. ./datasets if that's where "
                       "generate_trajectories.py downloaded them (default: the normal HF cache)")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--cache-location", default=None,
                  help="Where Hugging Face datasets are cached (e.g. ./datasets). Once the data is "
                       "there, set HF_DATASETS_OFFLINE=1 to run with no network.")
  ap.add_argument("--export", default=None,
                  help="Save the sampled questions to this JSON file and exit (no model needed). "
                       "Reuse later with --data FILE for a fully offline run.")
  ap.add_argument("adapters", nargs="*", help="path or path:max_new_tokens")
  ap.add_argument("--sequential", action="store_true")
  ap.add_argument("--batch-size", type=int, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--save", default=None)
  args = ap.parse_args()

  # Backward compatible: a .json file among the positionals is a question file, not an adapter.
  data_files = [args.data] if args.data else []
  adapter_args = []
  for a in args.adapters:
    (data_files if a.lower().endswith(".json") else adapter_args).append(a)
  if not adapter_args and not args.export:
    raise SystemExit("No adapters given. Example: python eval_adapters.py ./adapter_cot ./adapter_answer:24")
  args.adapters = adapter_args
  for path in (parse_adapter_arg(a)[0] for a in adapter_args):
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
      raise SystemExit(f"'{path}' is not a trained adapter folder (no adapter_config.json inside).")

  items = []
  for data_file in data_files:
    with open(data_file, "r", encoding="utf-8") as f:
      items.extend(json.load(f))
    print(f"Loaded {len(items)} questions from {data_file}")
  hf_spec = args.hf or (None if data_files else "arc:100,gsm8k:100")
  if hf_spec:
    for name, count in parse_hf_spec(hf_spec):
      got = HF_LOADERS[name](count, args.seed, args.cache_dir)
      print(f"Loaded {len(got)} {name} test questions from Hugging Face")
      items.extend(got)
  if not items:
    raise SystemExit("No questions to evaluate.")
  if args.limit:
    items = items[: args.limit]
  if args.export:
    with open(args.export, "w", encoding="utf-8") as f:
      json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(items)} questions to {args.export}. Use it with: --data {args.export}")
    return

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

  batch_size = args.batch_size or (8 if args.sequential else 24)

  def make_result(j, text, n_tok):
    pred = extract_answer(text)
    return {
        "index": j,
        "domain": items[j].get("domain", "all"),
        "gold": items[j]["ground_truth"],
        "pred": pred,
        "correct": answers_match(pred, items[j]["ground_truth"]),
        "generated_tokens": n_tok,
        "generation": text,
    }

  all_results = {path: [] for path, _ in specs}
  t0 = time.time()

  if args.sequential:
    for i, (path, max_new) in enumerate(specs):
      model.set_adapter(f"adapter_{i}")
      print(f"\n=== {path}  (max_new_tokens={max_new}) ===")
      order = sorted(range(len(items)), key=lambda j: len(prompts[j]))
      for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        texts, counts = generate_batch(model, tokenizer, [prompts[j] for j in idx], max_new)
        for j, text, n_tok in zip(idx, texts, counts):
          all_results[path].append(make_result(j, text, n_tok))
        print(f"  {min(start + batch_size, len(order))}/{len(order)}", end="\r")
      print()
  else:
    # One job per (adapter, question). Jobs with the same token budget are batched
    # together, ordered by prompt length, so different adapters share each batch.
    jobs = [(i, j) for i in range(len(specs)) for j in range(len(items))]
    jobs.sort(key=lambda ij: (specs[ij[0]][1], len(prompts[ij[1]]), ij[1], ij[0]))
    print(f"\nMixed-adapter mode: {len(jobs)} generations in batches of {batch_size}")
    for start in range(0, len(jobs), batch_size):
      chunk = jobs[start : start + batch_size]
      budget = max(specs[i][1] for i, _ in chunk)
      texts, counts = generate_batch(
          model, tokenizer,
          [prompts[j] for _, j in chunk],
          budget,
          adapter_names=[f"adapter_{i}" for i, _ in chunk],
      )
      for (i, j), text, n_tok in zip(chunk, texts, counts):
        all_results[specs[i][0]].append(make_result(j, text, n_tok))
      print(f"  {min(start + batch_size, len(jobs))}/{len(jobs)}", end="\r")
    print()
  print(f"Generation took {time.time() - t0:.0f}s")

  summary = []
  for path, _ in specs:
    results = sorted(all_results[path], key=lambda r: r["index"])
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
    print(f"\n{path}")
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
