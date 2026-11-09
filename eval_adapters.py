"""Score one or more Pythia-1B LoRA adapters on held-out questions.

Usage:
  # default: 100 random held-out questions from EACH of the 11 training domains
  # (arc, commonsense, deepmath, gsm8k, hotpot, logic_mcq, logiqa, math, mmlu, numina_pro, strategyqa)
  python eval_adapters.py ./adapter_answer:24 ./adapter_cot ./adapter_think

  # one number for every dataset
  python eval_adapters.py --n 50 ./adapter_cot

  # or per-dataset counts / a subset of datasets
  python eval_adapters.py --hf arc:200,gsm8k:50 ./adapter_cot

  # skip questions the adapters trained on (needed for math, deepmath, numina_pro, logic_mcq,
  # which only have a train split)
  python eval_adapters.py --exclude data_cot.json ./adapter_cot
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

LLM judge (optional, same idea as generate_trajectories.py):
  Answers are checked mechanically first (letters, yes/no, numbers by value). Only answers
  that check can't decide (free text, symbolic math) go to a judge model on an
  OpenAI-compatible server. Add:  --judge-model-label NAME  [--judge-host H --judge-port P]
  To judge after the fact, with no GPU model loaded:
      python eval_adapters_hf.py --rejudge results.json --judge-model-label NAME --judge-port 8081
  (results saved from now on include the problem text, which the judge needs.)

Speed options:
  --merge           merge each adapter into the base weights and run it alone, at plain-model
                    speed (no LoRA overhead per layer). Adapters are processed one at a time.
                    Merging in bf16/fp16 rounds weights slightly, so scores can differ by a
                    question or two from the unmerged modes.
  --no-loop-stop    by default a sequence is cut off once its last 16 tokens have already
                    appeared 4+ times (it is stuck repeating itself). Tune with --loop-ngram
                    and --loop-repeats. The LOOPED column counts how many were cut off.
  --no-early-stop   by default each sequence stops as soon as it has written a complete
                    'FINAL ANSWER: ...' line, so a model that forgets to stop can't hold up
                    its whole batch. This turns that off.

Options:
  --n N            questions per dataset (default 100)
  --cache-dir DIR  Hugging Face cache location
  --hf SPEC        datasets, comma separated, each optionally with :count (arc, commonsense, deepmath, gsm8k, hotpot,
                   logic_mcq, logiqa, math, mmlu, numina_pro, strategyqa).
                   Questions come from each dataset's TEST split, which the trajectory
                   generator never used (it read the train splits). Seeded, so every
                   run and every adapter sees the same questions.
  --data FILE      held-out JSON with problem / ground_truth (optional)
  --seed N         shuffle seed for --hf sampling (default 42)
  --exclude FILE   training JSON(s), comma-separated or repeated; their questions are never used
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
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from collections import Counter, defaultdict

try:
  import torch
  from peft import PeftModel
  from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
except ImportError:
  if "--rejudge" not in sys.argv:  # re-scoring a saved file needs no model libraries
    raise
  torch = None
  StoppingCriteria = object

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
def _strip_latex(t):
  t = re.sub(r"\\(?:left|right)\b", "", t)
  t = re.sub(r"\\[,;:! ]", "", t)
  t = re.sub(r"\\(?:text|textbf|mathrm|mathbf|boxed)\s*\{([^{}]*)\}", r"\1", t)
  return t.replace("$", "").strip()


def _to_fraction(t):
  """'72', '1,234', '0.75', '3/4', '\\frac{3}{4}' -> Fraction, else None."""
  s = _strip_latex(t).replace(",", "").replace(" ", "")
  m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
  if m:
    return Fraction(s)
  m = re.fullmatch(r"(-?)\\d?frac\{(\d+)\}\{(\d+)\}", s)
  if m and int(m.group(3)):
    f = Fraction(int(m.group(2)), int(m.group(3)))
    return -f if m.group(1) else f
  m = re.fullmatch(r"(-?\d+)/(\d+)", s)
  if m and int(m.group(2)):
    return Fraction(int(m.group(1)), int(m.group(2)))
  return None


def normalize(text):
  t = _strip_latex(text.strip()).rstrip(".").strip()
  m = re.fullmatch(r"\(?([A-Za-z])\)?", t)
  if m:  # option letter
    return m.group(1).upper()
  f = _to_fraction(t)
  if f is not None:  # numbers compare by value: 72 == 72.0 == $72
    return str(f)
  t = re.sub(r"\b(?:a|an|the)\b", " ", t.lower())  # HotpotQA/SQuAD-style exact match
  t = re.sub(r"[.,;:!?'\"]", "", t)
  return re.sub(r"\s+", " ", t).strip()


def extract_answer(generation):
  matches = FINAL_RE.findall(generation)
  return matches[-1].strip() if matches else None


def answers_match(pred, gold):
  return pred is not None and normalize(pred) == normalize(gold)


# ---------------------------------------------------------------------------
# Evaluation questions: every domain the adapters trained on, formatted by the
# generator's own extractors (copied verbatim) so prompts match training.
# ---------------------------------------------------------------------------
LETTERS = "ABCDEFGHIJ"


def format_mcq(question, options):
    body = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    return f"{question}\n\n{body}\n\nChoose the single best option."


def extract_all_boxed(text):
    """Contents of every \\boxed{...} in text, handling nested braces
    (\\frac{3}{4}, \\sqrt{2}, \\text{...})."""
    found, pos = [], 0
    while True:
        idx = text.find("\\boxed", pos)
        if idx == -1:
            return found
        j = idx + len("\\boxed")
        while j < len(text) and text[j].isspace():
            j += 1
        if j < len(text) and text[j] == "{":
            depth = 0
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        found.append(text[j + 1:k].strip())
                        pos = k + 1
                        break
            else:
                return found            # unbalanced braces -- stop
        else:
            pos = j


def looks_multipart(problem):
    """True for problems asking several things at once ("1. Find ... 2. Determine ...",
    "(1) ... (2) ...", "(I) ... (II) ..."). Their reference solutions box one answer
    per part, so no single expected answer exists."""
    if re.search(r"(?m)^\s*(?:\(1\)\.?|1[.)])\s", problem) and re.search(r"(?m)^\s*(?:\(2\)\.?|2[.)])\s", problem):
        return True
    if re.search(r"\(1\)\s+(?!and\b|or\b)[A-Za-z$\\].*?\(2\)", problem, flags=re.DOTALL):
        return True
    return bool(re.search(r"\((?:I|\u2160)\)[\s\S]*\((?:II|\u2161)\)", problem))


def is_checkable_answer(ground_truth):
    """Filters out ground truths no checker can fairly verify: proofs and
    equations ("DE \\cdot DF = DA^2"), combined multiple-choice letters
    ("BC"), and long prose answers. Keeps numbers, expressions, single
    letters, and short phrases."""
    g = ground_truth.strip()
    if not g or len(g) > 40:
        return False
    if re.fullmatch(r"[A-E]{2,}", g):
        return False
    if "=" in re.sub(r"^\s*[a-zA-Z]\s*=\s*", "", g):  # allow a bare "x = 5", reject real equations
        return False
    if len(re.findall(r"[A-Za-z]{3,}", re.sub(r"\\[A-Za-z]+", "", g))) > 3:
        return False
    return True


def extract_math(example):
    """Skips problems with no SINGLE verifiable answer: several different \\boxed answers
    in the reference solution, or a multi-part question. Otherwise a model that correctly
    answers every part gets graded against just the last one (and one that answers only
    the last part looks perfect). Skipped problems count as 'no extractable answer'."""
    problem, solution = example["problem"], example["solution"]
    boxes = extract_all_boxed(solution)
    if not boxes or len(set(boxes)) > 1 or looks_multipart(problem):
        return problem, None
    return problem, boxes[-1]


def extract_gsm8k(example):
    """openai/gsm8k -- 'answer' ends with '#### <number>'."""
    match = re.search(r"####\s*(.+)", example.get("answer", ""))
    ground_truth = match.group(1).strip().replace(",", "") if match else None
    return example.get("question"), ground_truth


def extract_logic_mcq(example):
    """hivaze/LOGIC-701 (en) -- 5-option multiple choice logic puzzles. Schema confirmed
    with --peek on the real data:
        problem_statement      the question
        answer_option_1..5     the options (some may be empty)
        correct_option_number  1-based index of the right option
        solution               a worked EXPLANATION -- not the answer; never ground truth
    (An earlier version guessed 'question' / 'solution', so the model saw only the options
    and was graded against the explanation text.)"""
    question = example.get("problem_statement")
    options = [(n, example.get(f"answer_option_{n}")) for n in range(1, 6)]
    options = [(n, text) for n, text in options if text]
    try:
        correct = int(example.get("correct_option_number"))
    except (TypeError, ValueError):
        return None, None
    numbers = [n for n, _ in options]
    if not question or correct not in numbers:
        return None, None
    return format_mcq(question, [text for _, text in options]), LETTERS[numbers.index(correct)]


def extract_logiqa(example):
    """lucasmccabe/logiqa -- context + query + 4 options, 'correct_option'
    is a 0-based index. PEEK FIRST."""
    options = example.get("options") or []
    idx = example.get("correct_option")
    if not options or idx is None:
        return None, None
    question = f"{example.get('context', '')}\n\n{example.get('query', '')}".strip()
    return format_mcq(question, options), LETTERS[int(idx)]


def _mcq_from_label_choices(example):
    """Shared by ARC and CommonsenseQA: choices = {'label': [...], 'text': [...]},
    answerKey = one of the labels. Re-letters options A, B, C... so every
    MCQ domain uses the same convention."""
    choices = example.get("choices") or {}
    labels, texts = list(choices.get("label", [])), list(choices.get("text", []))
    key = example.get("answerKey")
    if not texts or key not in labels:
        return None, None
    return format_mcq(example["question"], texts), LETTERS[labels.index(key)]


def extract_mmlu(example):
    """cais/mmlu -- 'choices' is a list, 'answer' is a 0-based index."""
    choices = example.get("choices") or []
    idx = example.get("answer")
    if not choices or idx is None:
        return None, None
    return format_mcq(example["question"], choices), LETTERS[int(idx)]


def extract_strategyqa(example):
    """ChilleD/StrategyQA -- yes/no questions needing implicit multi-step
    reasoning. 'answer' is a bool. PEEK FIRST."""
    if "answer" not in example or "question" not in example:
        return None, None
    return f"{example['question']}\n\nAnswer yes or no.", ("yes" if example["answer"] else "no")


def extract_hotpot(example):
    """hotpotqa/hotpot_qa -- multi-hop QA. Only the QUESTION is given to the
    model (not the supporting paragraphs), so it has to search -- that's the
    point of this domain. Free-text answers, so run with a judge for best
    yield. PEEK FIRST -- depends on the repo's parquet layout/config."""
    return example.get("question"), example.get("answer")


def extract_deepmath(example):
    """zwhe99/DeepMath-103K (MIT) -- 'question' + 'final_answer' (difficulty 3-10, mostly 5-9).
    The three r1_solution_* columns are R1 outputs and are deliberately never read."""
    question, answer = example.get("question"), example.get("final_answer")
    if not question or answer in (None, ""):
        return None, None
    return question, str(answer).strip()


def extract_numina_pro(example):
    """Azure99/NuminaMath-1.5-Pro (Apache-2.0). Only the problem and its reference 'answer' are
    used -- the regenerated solutions are ignored. If the row carries a 'hard' flag, easy rows
    are skipped. NuminaMath-1.5 marks unanswerable rows with 'proof' / 'notfound'; those are
    dropped. PEEK FIRST -- field names are from the card, row count was not checked."""
    if "hard" in example and not example.get("hard"):
        return None, None
    problem, answer = example.get("problem"), example.get("answer")
    if not problem or answer in (None, ""):
        return None, None
    answer = str(answer).strip()
    if answer.lower() in ("proof", "notfound", "not found", "n/a"):
        return None, None
    if looks_multipart(problem):
        return None, None
    return problem, answer


# Where each domain's EVAL questions come from. Wherever the dataset has a split the
# generator did not train from (it read train, and MMLU's test), that split is used.
# Domains marked "train only" have no other split: pass --exclude <training json> so
# questions the adapters trained on are skipped.
EVAL_DOMAINS = {
    "arc": dict(dataset="allenai/ai2_arc", config="ARC-Challenge", split="test",
                extract=_mcq_from_label_choices),
    "commonsense": dict(dataset="tau/commonsense_qa", config=None, split="validation",
                        extract=_mcq_from_label_choices),
    "deepmath": dict(dataset="zwhe99/DeepMath-103K", config=None, split="train",
                     extract=extract_deepmath, gt_filter=is_checkable_answer, stream=True),
    "gsm8k": dict(dataset="openai/gsm8k", config="main", split="test", extract=extract_gsm8k),
    "hotpot": dict(dataset="hotpotqa/hotpot_qa", config="distractor", split="validation",
                   extract=extract_hotpot),
    "logic_mcq": dict(dataset="hivaze/LOGIC-701", config="en", split="train",
                      extract=extract_logic_mcq),
    "logiqa": dict(dataset="lucasmccabe/logiqa", config=None, split="test", extract=extract_logiqa),
    "math": dict(dataset="AI-MO/NuminaMath-CoT", config=None, split="train",
                 extract=extract_math, gt_filter=is_checkable_answer, stream=True),
    "mmlu": dict(dataset="cais/mmlu", config="all", split="validation", extract=extract_mmlu),
    "numina_pro": dict(dataset="Azure99/NuminaMath-1.5-Pro", config=None, split="train",
                       extract=extract_numina_pro, gt_filter=is_checkable_answer, stream=True),
    "strategyqa": dict(dataset="ChilleD/StrategyQA", config=None, split="test",
                       extract=extract_strategyqa),
}


def _load(cfg, cache_dir, streaming):
  from datasets import load_dataset
  args = [cfg["dataset"]] + ([cfg["config"]] if cfg.get("config") else [])
  kwargs = dict(split=cfg["split"], streaming=streaming, cache_dir=cache_dir)
  try:
    return load_dataset(*args, **kwargs)
  except RuntimeError as e:  # newer `datasets` refuses dataset loading scripts; use the parquet copy
    if "scripts are no longer supported" not in str(e):
      raise
    return load_dataset(*args, revision="refs/convert/parquet", **kwargs)


def load_domain(name, n, seed, cache_dir=None, exclude=frozenset()):
  cfg = EVAL_DOMAINS[name]
  stream = cfg.get("stream", False)
  ds = _load(cfg, cache_dir, stream)
  ds = ds.shuffle(seed=seed, buffer_size=10_000) if stream else ds.shuffle(seed=seed)
  gt_filter = cfg.get("gt_filter")
  out = []
  for ex in ds:
    problem, truth = cfg["extract"](ex)
    if not problem or truth in (None, ""):
      continue
    problem, truth = problem.strip(), str(truth).strip()
    if gt_filter and not gt_filter(truth):
      continue
    if problem in exclude:
      continue
    out.append({"problem": problem, "ground_truth": truth, "domain": name})
    if len(out) >= n:
      break
  return out


def load_exclude(specs):
  """Problem texts from training JSON files (comma-separated paths, repeatable)."""
  seen = set()
  for spec in specs or []:
    for path in (p.strip() for p in spec.split(",") if p.strip()):
      with open(path, "r", encoding="utf-8") as f:
        seen.update(row["problem"].strip() for row in json.load(f))
  return seen


def parse_hf_spec(spec, default_count=100):
  pairs = []
  for part in spec.split(","):
    part = part.strip()
    if not part:
      continue
    name, _, count = part.partition(":")
    if name not in EVAL_DOMAINS:
      raise SystemExit(f"Unknown dataset '{name}'. Available: {', '.join(EVAL_DOMAINS)}")
    pairs.append((name, int(count) if count else default_count))
  return pairs


# ---------------------------------------------------------------------------
# Judge helpers copied verbatim from generate_trajectories.py
# ---------------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Known correct answer: {ground_truth}\n"
    "Candidate answer: {candidate}\n\n"
    "Does the candidate answer mean the same thing as the known correct "
    "answer, ignoring formatting, units, phrasing, or notation differences? "
    "Respond with exactly one word: YES or NO."
)


def canonicalize(kind, answer):
    if answer is None:
        return None
    a = answer.strip().strip("*. ")
    if kind == "mcq":
        m = re.match(r"^\(?([A-Fa-f])\)?(?:[.):\s]|$)", a)
        return m.group(1).upper() if m else a
    if kind == "yesno":
        low = a.lower()
        if re.match(r"^(yes|true)\b", low):
            return "yes"
        if re.match(r"^(no|false)\b", low):
            return "no"
        return a
    return a


def parse_judge_verdict(text):
    """YES -> True, NO -> False, anything else -> None (no usable verdict).
    Inline <think> blocks are stripped first, since some servers leave a thinking
    model's reasoning in the reply -- and it often contains stray yes/no words."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)  # cut off mid-thought
    match = re.search(r"\b(YES|NO)\b", text.upper())
    return None if match is None else match.group(1) == "YES"


def build_base_url(host, port):
    """'localhost' + 8080 -> http://localhost:8080/v1. A full URL (for remote
    or HTTPS endpoints) is used as given, with /v1 appended if it's missing;
    --port is ignored in that case. A host that already carries its own port
    ('192.168.1.20:8081') is also used as given."""
    host = (host or "localhost").strip()
    if host.startswith(("http://", "https://")):
        url = host.rstrip("/")
        return url if url.endswith("/v1") else url + "/v1"
    if re.search(r":\d+$", host):
        return f"http://{host}/v1"
    return f"http://{host}:{port}/v1"


# ---------------------------------------------------------------------------
# Verdicts: mechanical first, LLM judge only for what it can't decide
# (same strategy as verify_answer in generate_trajectories.py)
# ---------------------------------------------------------------------------
DOMAIN_KIND = {"arc": "mcq", "commonsense": "mcq", "logic_mcq": "mcq", "logiqa": "mcq",
               "mmlu": "mcq", "strategyqa": "yesno"}  # everything else is free-form "text"


def kind_of(domain):
  return DOMAIN_KIND.get(domain, "text")


def mechanical_verdict(pred, gold, kind):
  """True = same answer. False = decisively different (two letters, yes vs no, two plain
  numbers) or no answer at all. None = can't tell mechanically, worth asking a judge."""
  if pred is None:
    return False
  p, g = canonicalize(kind, pred), canonicalize(kind, gold)
  np_, ng = normalize(p), normalize(g)
  if np_ == ng:
    return True
  if re.fullmatch(r"[A-Za-z]", np_) and re.fullmatch(r"[A-Za-z]", ng):
    return False
  if {np_, ng} == {"yes", "no"}:
    return False
  if _to_fraction(p) is not None and _to_fraction(g) is not None:
    return False
  return None


def score_result(r):
  """(Re)computes the mechanical verdict; clears any earlier judge verdict."""
  mech = mechanical_verdict(r["pred"], r["gold"], kind_of(r["domain"]))
  r["mechanical"] = mech
  r["correct"] = mech is True
  r["method"] = ("none" if r["pred"] is None else
                 "mechanical" if mech is True else
                 "mechanical_mismatch" if mech is False else "unjudged")
  r.pop("judge", None)


def make_judge_cfg(args):
  if not args.judge_model_label:
    return None
  return dict(
      model=args.judge_model_label,
      url=build_base_url(args.judge_host, args.judge_port),
      timeout=args.judge_timeout,
      max_tokens=args.judge_max_tokens,
      api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
      workers=args.judge_workers,
  )


def llm_judge_match(candidate, gold, problem, cfg):
  """True / False, or None when there's no usable verdict (call failed or no YES/NO)."""
  prompt = JUDGE_PROMPT_TEMPLATE.format(problem=problem, ground_truth=gold, candidate=candidate)
  body = {
      "model": cfg["model"],
      "messages": [{"role": "user", "content": prompt}],
      "temperature": 0,
      "max_tokens": cfg["max_tokens"],
      "chat_template_kwargs": {"enable_thinking": False},  # a one-word verdict needs no reasoning chain
  }
  req = urllib.request.Request(
      cfg["url"] + "/chat/completions",
      data=json.dumps(body).encode("utf-8"),
      headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"},
      method="POST")
  try:
    with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
      choice = json.loads(resp.read().decode("utf-8"))["choices"][0]
    verdict = parse_judge_verdict((choice.get("message") or {}).get("content"))
    if verdict is None:
      why = ("ran out of tokens before answering -- raise --judge-max-tokens"
             if choice.get("finish_reason") == "length" else "reply had no YES/NO in it")
      print(f"    (judge gave no usable verdict: {why})")
    return verdict
  except Exception as e:
    hint = " -- try a larger --judge-timeout" if "timed out" in str(e).lower() else ""
    print(f"    (judge call failed: {e}{hint})")
    return None


def run_judge(all_results, cfg):
  todo = [r for rs in all_results.values() for r in rs if r["mechanical"] is None]
  print(f"\nJudge: {cfg['model']} at {cfg['url']} -- {len(todo)} answers the mechanical check couldn't decide")
  if not todo:
    return
  if any(not r.get("problem") for r in todo):
    print("  warning: some results were saved without the problem text, so the judge sees only the answers")
  done = 0
  with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
    futures = {pool.submit(llm_judge_match, r["pred"], r["gold"], r.get("problem", ""), cfg): r
               for r in todo}
    for fut in as_completed(futures):
      r, verdict = futures[fut], fut.result()
      r["judge"] = verdict
      if verdict is None:
        r["method"] = "judge_error"
      else:
        r["correct"] = verdict
        r["method"] = "llm_judge" if verdict else "judge_no"
      done += 1
      print(f"  judged {done}/{len(todo)}", end="\r")
  print()


def report(all_results):
  labels = list(all_results)
  first = all_results[labels[0]]
  golds = [normalize(r["gold"]) for r in first]
  majority = Counter(golds).most_common(1)[0][1] / len(golds)
  judged_any = any("judge" in r for rs in all_results.values() for r in rs)

  print("\n" + "=" * 98)
  head = f"{'ADAPTER':<30} {'ACC':>6} {'CORRECT':>9} "
  if judged_any:
    head += f"{'MECH ONLY':>10} {'JUDGED':>7} {'JUDGE YES':>10} "
  print(head + f"{'NO FINAL':>9} {'LOOPED':>7} {'AVG TOK':>8}")
  for label in labels:
    rs = all_results[label]
    n = len(rs)
    correct = sum(r["correct"] for r in rs)
    line = f"{label:<30} {correct / n:>6.0%} {f'{correct}/{n}':>9} "
    if judged_any:
      mech = sum(r["mechanical"] is True for r in rs)
      judged = sum(r.get("judge") is not None for r in rs)
      yes = sum(r.get("judge") is True for r in rs)
      line += f"{mech:>10} {judged:>7} {yes:>10} "
    no_final = sum(r["pred"] is None for r in rs)
    avg_tok = sum(r["generated_tokens"] for r in rs) / n
    looped = sum(bool(r.get("looped")) for r in rs)
    print(line + f"{no_final:>9} {looped:>7} {avg_tok:>8.0f}")
  print(f"{'(most-common-answer baseline)':<30} {majority:>6.0%}")

  domains = sorted({r["domain"] for r in first})
  stats = {lb: defaultdict(lambda: [0, 0]) for lb in labels}
  for lb in labels:
    for r in all_results[lb]:
      stats[lb][r["domain"]][0] += r["correct"]
      stats[lb][r["domain"]][1] += 1
  names = [os.path.basename(lb.rstrip("/"))[:16] for lb in labels]
  print("\nPER-DOMAIN ACCURACY  (guess* = share of the single most common answer)")
  print(f"{'domain':<13}{'n':>5}{'guess*':>8}" + "".join(f"{nm:>18}" for nm in names))
  for d in domains:
    dgolds = [normalize(r["gold"]) for r in first if r["domain"] == d]
    guess = Counter(dgolds).most_common(1)[0][1] / len(dgolds)
    cells = "".join(
        f"{f'{stats[lb][d][0]}/{stats[lb][d][1]} ({stats[lb][d][0] / max(1, stats[lb][d][1]):.0%})':>18}"
        for lb in labels)
    print(f"{d:<13}{len(dgolds):>5}{guess:>8.0%}{cells}")


def rejudge_main(args):
  """Re-score (and optionally judge) a --save file without loading any model."""
  with open(args.rejudge, "r", encoding="utf-8") as f:
    all_results = json.load(f)
  for rs in all_results.values():
    for r in rs:
      score_result(r)
  cfg = make_judge_cfg(args)
  if cfg:
    run_judge(all_results, cfg)
  else:
    print("No --judge-model-label given: mechanical scoring only.")
  report(all_results)
  out = args.save or re.sub(r"\.json$", "", args.rejudge) + ".judged.json"
  with open(out, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
  print(f"\nSaved re-scored results to {out}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
EARLY_STOP = True
FINAL_DONE_RE = re.compile(r"FINAL ANSWER:[^\n]+\n")


class FinalAnswerStop(StoppingCriteria):
  """Marks a row finished once it has written a complete 'FINAL ANSWER: ...' line.

  Checked every few steps to keep CPU overhead low; rows already finished stay finished.
  """

  def __init__(self, tokenizer, prompt_len, every=4, window=48):
    self.tokenizer, self.prompt_len, self.every, self.window = tokenizer, prompt_len, every, window
    self.step = 0

  def __call__(self, input_ids, scores, **kwargs):
    self.step += 1
    done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    if self.step % self.every:
      return done
    start = max(self.prompt_len, input_ids.shape[1] - self.window)
    tails = self.tokenizer.batch_decode(input_ids[:, start:], skip_special_tokens=True)
    for r, tail in enumerate(tails):
      if FINAL_DONE_RE.search(tail):
        done[r] = True
    return done


LOOP_STOP = True
LOOP_NGRAM = 16
LOOP_REPEATS = 4


class LoopStop(StoppingCriteria):
  """Ends a row once its output is stuck in a loop.

  A row is looping when its last LOOP_NGRAM tokens already appeared at least LOOP_REPEATS
  times in what it has generated so far (e.g. 'Let's use calculator.' over and over).
  Works on token ids, is checked every few steps, and ignores rows that already finished
  (their tail is padding). self.looped[row] records which rows were cut off.
  """

  def __init__(self, prompt_len, n_rows, finished_ids, ngram, repeats, every=8, min_tokens=64):
    self.prompt_len, self.finished_ids = prompt_len, set(finished_ids)
    self.ngram, self.repeats, self.every, self.min_tokens = ngram, repeats, every, min_tokens
    self.looped = [False] * n_rows
    self.step = 0

  def __call__(self, input_ids, scores, **kwargs):
    self.step += 1
    done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    if self.step % self.every:
      return done
    n = self.ngram
    for r, toks in enumerate(input_ids[:, self.prompt_len:].tolist()):
      if len(toks) < self.min_tokens or toks[-1] in self.finished_ids:
        continue
      tail = toks[-n:]
      first, count = tail[0], 0
      for i in range(len(toks) - n + 1):
        if toks[i] == first and toks[i:i + n] == tail:
          count += 1
          if count >= self.repeats:
            done[r] = True
            self.looped[r] = True
            break
    return done


def generate_batch(model, tokenizer, prompts, max_new_tokens, adapter_names=None):
  enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
  prompt_len = enc["input_ids"].shape[1]
  budget = min(max_new_tokens, CONTEXT - prompt_len)
  if budget < 1:  # prompt alone fills the context window
    return [""] * len(prompts), [0] * len(prompts), [False] * len(prompts)

  criteria = []
  if EARLY_STOP:
    criteria.append(FinalAnswerStop(tokenizer, prompt_len))
  loop = None
  if LOOP_STOP:
    loop = LoopStop(prompt_len, len(prompts), (tokenizer.pad_token_id, tokenizer.eos_token_id),
                    LOOP_NGRAM, LOOP_REPEATS)
    criteria.append(loop)

  extra = {"adapter_names": adapter_names} if adapter_names is not None else {}
  with torch.no_grad():
    out = model.generate(
        **enc,
        **extra,
        max_new_tokens=budget,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **({"stopping_criteria": StoppingCriteriaList(criteria)} if criteria else {}),
    )
  gen = out[:, prompt_len:]
  texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
  counts = (gen != tokenizer.pad_token_id).sum(dim=1).tolist()
  return texts, counts, (loop.looped if loop else [False] * len(prompts))


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
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--exclude", action="append", default=[],
                  help="Training JSON(s) (e.g. data_cot.json) whose questions must not be used for eval. "
                       "Comma-separate or repeat. Needed for domains with no held-out split (math, deepmath, "
                       "numina_pro, logic_mcq).")
  ap.add_argument("--cache-dir", default=None,
                  help="Where Hugging Face datasets are cached (e.g. ./datasets). Once the data is "
                       "there, set HF_DATASETS_OFFLINE=1 to run with no network.")
  ap.add_argument("--export", default=None,
                  help="Save the sampled questions to this JSON file and exit (no model needed). "
                       "Reuse later with --data FILE for a fully offline run.")
  ap.add_argument("adapters", nargs="*", help="path or path:max_new_tokens")
  ap.add_argument("--sequential", action="store_true")
  ap.add_argument("--merge", action="store_true",
                  help="Merge each adapter into the base weights and run adapters one at a time (faster per token).")
  ap.add_argument("--no-loop-stop", action="store_true",
                  help="Don't cut off generations that are stuck repeating themselves.")
  ap.add_argument("--loop-ngram", type=int, default=16,
                  help="Loop test: length in tokens of the repeated chunk (default 16)")
  ap.add_argument("--loop-repeats", type=int, default=4,
                  help="Loop test: how many times that chunk must already have appeared (default 4)")
  ap.add_argument("--no-early-stop", action="store_true",
                  help="Don't stop sequences at the end of the FINAL ANSWER line.")
  ap.add_argument("--judge-model-label", default=None,
                  help="Turns on the LLM judge: the model name to ask on an OpenAI-compatible server "
                       "(e.g. llama-server). It only sees answers the mechanical check can't decide.")
  ap.add_argument("--judge-host", default="localhost", help="Judge server host or full URL")
  ap.add_argument("--judge-port", type=int, default=8080)
  ap.add_argument("--judge-timeout", type=float, default=180, help="Seconds per judge call")
  ap.add_argument("--judge-max-tokens", type=int, default=1024)
  ap.add_argument("--judge-workers", type=int, default=4, help="Concurrent judge requests")
  ap.add_argument("--api-key", default=None, help="Judge API key (default: $OPENAI_API_KEY or none)")
  ap.add_argument("--rejudge", default=None, metavar="RESULTS.json",
                  help="Re-score (and judge) a --save file. Loads no model, so you can stop Pythia and "
                       "start the judge server first.")
  ap.add_argument("--batch-size", type=int, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--save", default=None)
  args = ap.parse_args()
  global EARLY_STOP, LOOP_STOP, LOOP_NGRAM, LOOP_REPEATS
  EARLY_STOP = not args.no_early_stop
  LOOP_STOP = not args.no_loop_stop
  LOOP_NGRAM, LOOP_REPEATS = args.loop_ngram, args.loop_repeats
  if args.rejudge:
    rejudge_main(args)
    return

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
  hf_spec = args.hf or (None if data_files else ",".join(EVAL_DOMAINS))
  if hf_spec:
    exclude = load_exclude(args.exclude)
    if exclude:
      print(f"Excluding {len(exclude)} training questions")
    for name, count in parse_hf_spec(hf_spec, args.n):
      split = EVAL_DOMAINS[name]["split"]
      if split == "train" and not exclude:
        print(f"  note: {name} only has a train split; pass --exclude <training json> to skip trained-on questions")
      try:
        got = load_domain(name, count, args.seed, args.cache_dir, exclude)
      except Exception as e:  # one broken dataset shouldn't sink the whole eval
        print(f"Skipping {name}: {type(e).__name__}: {str(e)[:200]}")
        continue
      print(f"Loaded {len(got)}/{count} {name} questions ({split} split)")
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

  def load_base():
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
    )

  specs = [parse_adapter_arg(a) for a in args.adapters]
  model = None
  if not args.merge:  # merge mode loads and merges one adapter at a time instead
    base = load_base()
    for i, (path, _) in enumerate(specs):
      name = f"adapter_{i}"
      if model is None:
        model = PeftModel.from_pretrained(base, path, adapter_name=name)
      else:
        model.load_adapter(path, adapter_name=name)
    model.eval()

  batch_size = args.batch_size or (8 if args.sequential else 32 if args.merge else 24)

  def make_result(j, text, n_tok, looped=False):
    r = {
        "index": j,
        "domain": items[j].get("domain", "all"),
        "problem": items[j]["problem"].strip(),
        "gold": items[j]["ground_truth"],
        "pred": extract_answer(text),
        "generated_tokens": n_tok,
        "looped": looped,
        "generation": text,
    }
    score_result(r)
    return r

  all_results = {path: [] for path, _ in specs}
  t0 = time.time()

  if args.merge:
    import gc
    for path, max_new in specs:
      print(f"\n=== {path}  (merged, max_new_tokens={max_new}) ===")
      merged = PeftModel.from_pretrained(load_base(), path).merge_and_unload()
      merged.eval()
      order = sorted(range(len(items)), key=lambda j: len(prompts[j]))
      for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        texts, counts, loops = generate_batch(merged, tokenizer, [prompts[j] for j in idx], max_new)
        for j, text, n_tok, looped in zip(idx, texts, counts, loops):
          all_results[path].append(make_result(j, text, n_tok, looped))
        print(f"  {min(start + batch_size, len(order))}/{len(order)}", end="\r")
      print()
      del merged
      gc.collect()
      torch.cuda.empty_cache()
  elif args.sequential:
    for i, (path, max_new) in enumerate(specs):
      model.set_adapter(f"adapter_{i}")
      print(f"\n=== {path}  (max_new_tokens={max_new}) ===")
      order = sorted(range(len(items)), key=lambda j: len(prompts[j]))
      for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        texts, counts, loops = generate_batch(model, tokenizer, [prompts[j] for j in idx], max_new)
        for j, text, n_tok, looped in zip(idx, texts, counts, loops):
          all_results[path].append(make_result(j, text, n_tok, looped))
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
      texts, counts, loops = generate_batch(
          model, tokenizer,
          [prompts[j] for _, j in chunk],
          budget,
          adapter_names=[f"adapter_{i}" for i, _ in chunk],
      )
      for (i, j), text, n_tok, looped in zip(chunk, texts, counts, loops):
        all_results[specs[i][0]].append(make_result(j, text, n_tok, looped))
      print(f"  {min(start + batch_size, len(jobs))}/{len(jobs)}", end="\r")
    print()
  print(f"Generation took {time.time() - t0:.0f}s")

  for path in all_results:
    all_results[path].sort(key=lambda r: r["index"])
  judge_cfg = make_judge_cfg(args)
  if judge_cfg:
    run_judge(all_results, judge_cfg)
  report(all_results)

  if args.save:
    with open(args.save, "w", encoding="utf-8") as f:
      json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved generations to {args.save}")


if __name__ == "__main__":
  main()
