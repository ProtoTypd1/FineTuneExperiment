"""Convert generate_trajectories.py / pool_generate.py output into flat training JSON.

Input: a .jsonl file, or a directory (searched recursively for *.jsonl).

Outputs (same kept rows in each):
  <out>_cot.json     problem, ground_truth, response   visible written reasoning, with tool calls/results
                                                       rendered inline, ending in "FINAL ANSWER: X"
  <out>_think.json   same, but each turn's hidden reasoning_content is prepended as <think>...</think>
                     (only differs from _cot for rows generated with thinking enabled)
  <out>_answer.json  problem, ground_truth             target is just "FINAL ANSWER: X"
  <out>_heldout.json problem, ground_truth, domain     questions removed from all three files above,
                                                       for eval_adapters.py (split by problem, seeded)

Usage:
  python prepare_data.py ./output/ --out data --verified-only
  python prepare_data.py trajectories_verified_xxx_gsm8k.jsonl --out data
"""
import argparse
import json
import pathlib
import random
import re
from collections import Counter

FINAL_RE = re.compile(r"FINAL ANSWER:\s*", re.IGNORECASE)

# Tool failures that say nothing about the problem (environment noise).
TOOL_NOISE = [
    "could not connect to docker",
    "sandbox is unavailable",
    "is not available in this session",
    "tool is unavailable",
]


def render_tool_call(tc, results):
  fn = tc.get("function", {})
  name = fn.get("name", "tool")
  try:
    args = json.loads(fn.get("arguments") or "{}")
  except Exception:
    args = {}
  if not isinstance(args, dict):
    args = {}
  op = args.pop("operation", None)
  label = f"{name}.{op}" if op else name
  if len(args) == 1:
    arg_text = str(next(iter(args.values())))
  else:
    arg_text = ", ".join(f"{k}={v}" for k, v in args.items())

  raw = results.get(tc.get("id"), "")
  try:
    res = json.loads(raw)
    out = str(res.get("result")) if res.get("ok") else f"error: {res.get('error')}"
  except Exception:
    out = raw
  out = out.replace("\n", " ")
  if len(out) > 300:
    out = out[:300] + "..."
  return f"[{label}({arg_text}) -> {out}]"


def has_tool_noise(messages):
  for m in messages:
    if m["role"] == "tool":
      low = (m.get("content") or "").lower()
      if any(p in low for p in TOOL_NOISE):
        return True
  return False


def build_response(messages, with_think):
  results = {m["tool_call_id"]: m.get("content") or "" for m in messages if m["role"] == "tool"}
  blocks = []
  for m in messages:
    if m["role"] != "assistant":
      continue
    lines = []
    thought = (m.get("reasoning_content") or "").strip()
    if with_think and thought:
      lines.append(f"<think>\n{thought}\n</think>")
    content = (m.get("content") or "").strip()
    if content:
      lines.append(content)
    for tc in m.get("tool_calls") or []:
      lines.append(render_tool_call(tc, results))
    if lines:
      blocks.append("\n".join(lines))
  return "\n\n".join(blocks).strip()


def set_final_line(response, final_answer):
  """Make the response end with exactly one canonical 'FINAL ANSWER: X' line."""
  matches = list(FINAL_RE.finditer(response))
  head = response[:matches[-1].start()].rstrip() if matches else response.rstrip()
  return f"{head}\n\nFINAL ANSWER: {final_answer}"


def convert(row):
  """Return (sample, None) or (None, skip_reason)."""
  if row.get("reject_reason"):
    return None, f"rejected:{row['reject_reason']}"
  final_answer = (row.get("final_answer") or "").strip()
  if not final_answer:
    return None, "no_final_answer"
  if (row.get("usage") or {}).get("segments", 1) > 1 or row.get("segment_starts"):
    return None, "rolled_over"

  msgs = row.get("messages", [])
  user = next((m for m in msgs if m["role"] == "user"), None)
  if user is None:
    return None, "no_user_message"
  if has_tool_noise(msgs):
    return None, "tool_noise"

  cot = build_response(msgs, with_think=False)
  if not cot:
    return None, "empty_response"
  think = build_response(msgs, with_think=True)

  return {
      "problem": user["content"].strip(),
      "ground_truth": final_answer,  # generator-verified, canonicalized form
      "response_cot": set_final_line(cot, final_answer),
      "response_think": set_final_line(think, final_answer),
      "domain": row.get("domain", "unknown"),
      "thinking_enabled": row.get("thinking_enabled", False),
      "source_index": row.get("source_index"),
  }, None


def find_jsonl(root, verified_only):
  files = sorted(root.rglob("*.jsonl"))
  if verified_only:
    files = [f for f in files if "verified" in f.name.lower()]
  return files


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("input", help="A .jsonl file or a directory.")
  ap.add_argument("--out", default="data")
  ap.add_argument("--verified-only", action="store_true",
                  help="In a directory, only read files with 'verified' in the name.")
  ap.add_argument("--holdout", type=float, default=0.1,
                  help="Fraction of unique problems (per domain) held out for evaluation. 0 disables.")
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  root = pathlib.Path(args.input)
  files = find_jsonl(root, args.verified_only) if root.is_dir() else [root]
  if not files:
    raise SystemExit(f"No *.jsonl files found under {root}")

  samples, stats, total = [], {}, 0
  for path in files:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    total += len(rows)
    kept, skipped = 0, Counter()
    for row in rows:
      s, reason = convert(row)
      if s is None:
        skipped[reason] += 1
      else:
        samples.append(s)
        kept += 1
    stats[path.name] = (len(rows), kept, skipped)

  # Hold out whole problems (by problem text) so no variant trains on an eval question.
  rng = random.Random(args.seed)
  by_domain = {}
  for s in samples:
    by_domain.setdefault(s["domain"], set()).add(s["problem"])
  held = set()
  for probs in by_domain.values():
    uniq = sorted(probs)
    rng.shuffle(uniq)
    held.update(uniq[:int(round(args.holdout * len(uniq)))])
  heldout, seen = [], set()
  for s in samples:
    if s["problem"] in held and s["problem"] not in seen:
      seen.add(s["problem"])
      heldout.append({"problem": s["problem"], "ground_truth": s["ground_truth"],
                      "domain": s["domain"], "source_index": s["source_index"]})
  samples = [s for s in samples if s["problem"] not in held]

  meta = ("domain", "thinking_enabled", "source_index")
  def dump(suffix, items):
    p = pathlib.Path(f"{args.out}_{suffix}.json")
    with p.open("w", encoding="utf-8") as f:
      json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Wrote {p} ({len(items)} samples)")

  dump("cot", [{"problem": s["problem"], "ground_truth": s["ground_truth"],
                "response": s["response_cot"], **{k: s[k] for k in meta}} for s in samples])
  dump("think", [{"problem": s["problem"], "ground_truth": s["ground_truth"],
                  "response": s["response_think"], **{k: s[k] for k in meta}} for s in samples])
  dump("answer", [{"problem": s["problem"], "ground_truth": s["ground_truth"],
                   **{k: s[k] for k in meta}} for s in samples])

  if heldout:
    dump("heldout", heldout)

  print(f"\n{'FILE':<52} {'READ':>5} {'KEPT':>5}  SKIPPED")
  for name, (n, k, sk) in stats.items():
    print(f"{name:<52} {n:>5} {k:>5}  {dict(sk) or '-'}")
  print(f"{'TOTAL':<52} {total:>5} {len(samples):>5}")
  print("domains:", dict(Counter(s["domain"] for s in samples)))


if __name__ == "__main__":
  main()
