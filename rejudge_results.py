#!/usr/bin/env python3
"""
rejudge_results.py  (standalone: imports only the Python standard library; sympy is optional)

Re-judges the records in an eval results.json whose method is "unjudged".
(`generate_trajectories.py --rejudge` works on trajectories_rejected_*.jsonl files,
not on this format, so this is the equivalent for results.json.)

Each unjudged record goes through the same two steps as the trajectory rejudge:
  1. free mechanical check   mechanical_match(pred, gold)   -> True / False / None
  2. LLM judge, only when step 1 can't decide   -> True / False / None
A judge failure leaves the record "unjudged" so a re-run retries it.

The matching and judging code below is copied from generate_trajectories.py so this
file runs on its own. If you change the matching rules there, change them here too.
Optional: `pip install sympy` for symbolic/numeric equivalence (e.g. 1/2 == 0.5).
Without it those cases are sent to the LLM judge instead of being settled for free.

The judge is any OpenAI-compatible server (llama.cpp, vLLM, ...); it is called over
plain HTTP, so the openai package is not needed.

Input is never modified. Output goes to --out (default results_rejudged.json);
if --out already exists it is used as the starting point, so an interrupted run
resumes instead of starting over. Each changed record keeps "original_method".

USAGE:
    # mechanical only, no server needed -- see how much it settles for free
    python rejudge_results.py --results results.json

    # with the LLM judge
    python rejudge_results.py --results results.json \\
        --judge-host localhost --judge-port 8081 --judge-model-label <judge-model-name>

    # a few judge calls at once, if your server can handle it
    python rejudge_results.py --results results.json --judge-port 8081 \\
        --judge-model-label <name> --workers 4
"""

import argparse
import json
import os
import re
import sys
import threading
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import sympy
except ImportError:          # optional -- see docstring
    sympy = None

JUDGE_TIMEOUT = 900          # seconds per judge call (--judge-timeout)
JUDGE_MAX_TOKENS = 16000     # cap on the judge's reply (--judge-max-tokens)

JUDGE_PROMPT_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Known correct answer: {ground_truth}\n"
    "Candidate answer: {candidate}\n\n"
    "Does the candidate answer mean the same thing as the known correct "
    "answer, ignoring formatting, units, phrasing, or notation differences? "
    "Respond with exactly one word: YES or NO."
)


# ---- matching code, copied verbatim from generate_trajectories.py ----

def normalize_answer(text):
    t = text.strip().strip("*. ")
    t = re.sub(r"\\boxed\{(.+?)\}", r"\1", t)
    t = re.sub(r"\\[()[\]]", "", t)
    t = t.strip("$").strip()
    t = re.sub(r"^\((\w)\)$", r"\1", t)
    t = t.replace("\\%", "%")
    # strip thousands separators ("12,345,678"), but NOT the comma in a tuple/interval:
    # the old blanket rule turned (-1,0) into (-10) and made (1,2) "equal" (12)
    t = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: m.group(0).replace(",", ""), t)
    t = re.sub(r"^x\s*=\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^(-?\d+(?:\.\d+)?)\s*(?:cents|dollars|frames|units|%)\b.*$", r"\1", t, flags=re.IGNORECASE)
    return t.strip()


_SAFE_EXPR = re.compile(r"[A-Za-z0-9\s+\-*/^().,]*")
_UNSAFE_NAMES = re.compile(r"\b(?:eval|exec|open|compile|input|getattr|setattr|delattr|globals|locals|vars|dir|"
                           r"chr|ord|exit|quit|help|breakpoint|import|lambda)\b")


def latex_to_expr(text):
    """Rewrites the LaTeX that shows up in answers (fractions, square roots, braced
    exponents, cdot/times, pi) into plain expression syntax a math parser understands."""
    t = re.sub(r"\\(?:left|right)\b", "", text)
    t = re.sub(r"\\[,;:! ]", "", t)
    t = t.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi")
    previous = None
    while previous != t:                        # innermost-first, so nesting works
        previous = t
        t = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", t)
        t = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", t)
        t = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", t)
    return t.replace("^", "**").replace("{", "(").replace("}", ")")


def to_expression(text):
    """Parses an answer into a sympy expression, or None if it isn't plain math.
    Model output is untrusted and the parser evaluates text, so only a narrow
    character set (no quotes, underscores or brackets) and no dangerous names get through."""
    if sympy is None:
        return None
    expr_text = latex_to_expr(text)
    if not expr_text.strip() or not _SAFE_EXPR.fullmatch(expr_text) or _UNSAFE_NAMES.search(expr_text):
        return None
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application)
        result = parse_expr(expr_text, transformations=standard_transformations +
                            (implicit_multiplication_application,))
    except Exception:
        return None
    return result if isinstance(result, sympy.Basic) else None


_ROLE_MAPPING = re.compile(r"\s*[^=,]+=\s*(?:knight|knave)(?:\s*,\s*[^=,]+=\s*(?:knight|knave))*\s*")


def parse_role_mapping(text):
    """'Zoey=knight, Oliver=knave' -> {'Zoey': 'knight', 'Oliver': 'knave'}; None if text isn't in
    that canonical form (so only knights-and-knaves ground truths take this path)."""
    if not isinstance(text, str) or not _ROLE_MAPPING.fullmatch(text):
        return None
    pairs = (part.split("=") for part in text.split(","))
    return {name.strip(): role.strip().lower() for name, role in pairs}


def parse_roles(names, text):
    """Reads {name: 'knight'|'knave'} for every name in `names` out of a free-text answer:
    'Zoey is a knight', 'Zoey = knave', 'Zoey: knight', 'Zoey is not a knight' (= knave),
    or role lists like 'Knights: Zoey, Harper. Knaves: Oliver'. Returns None unless the text
    settles EVERY name, once each, without contradicting itself."""
    text = text or ""
    seen = {name: set() for name in names}
    for name in names:
        boundary = rf"(?<![\w'])" + re.escape(name) + rf"(?![\w'])"
        pattern = boundary + r"\s*(?:is\s+(?:(not)\s+)?(?:an?\s+)?|[:=]\s*|-\s*|\(\s*)(knight|knave)s?\b"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            role = m.group(2).lower()
            if m.group(1):                                  # "is not a knight" -> knave (only two roles)
                role = "knave" if role == "knight" else "knight"
            seen[name].add(role)
    for m in re.finditer(r"(knight|knave)s?\s*[:=]\s*([^.;\n]*)", text, flags=re.IGNORECASE):
        for name in names:
            if re.search(rf"(?<![\w'])" + re.escape(name) + rf"(?![\w'])", m.group(2)):
                seen[name].add(m.group(1).lower())
    if any(len(roles) != 1 for roles in seen.values()):
        return None
    return {name: next(iter(roles)) for name, roles in seen.items()}


def mechanical_match(generated, ground_truth):
    """True  = same answer.
    False = DECISIVELY different (two different plain numbers, two different
            single letters, or yes vs no) -- no judge needed, the verdict is final.
    None  = can't tell mechanically (text, symbolic forms, tuples...) -- worth
            asking a judge."""
    if generated is None or ground_truth is None:
        return False

    truth_roles = parse_role_mapping(ground_truth)          # knights-and-knaves: compare who is what
    if truth_roles is not None:
        got = parse_roles(list(truth_roles), generated)
        return None if got is None else got == truth_roles

    g_norm = normalize_answer(generated)
    t_norm = normalize_answer(ground_truth)
    g_low, t_low = g_norm.strip().lower(), t_norm.strip().lower()

    if g_low == t_low or re.sub(r"\s+", "", g_low) == re.sub(r"\s+", "", t_low):
        return True
    if re.fullmatch(r"[a-z]", g_low) and re.fullmatch(r"[a-z]", t_low):        # MCQ letters
        return False
    if {g_low, t_low} == {"yes", "no"}:
        return False

    g, t = to_expression(g_norm), to_expression(t_norm)
    if g is None or t is None:
        return None
    try:
        if sympy.simplify(g - t) == 0:
            return True
        if not g.free_symbols and not t.free_symbols:                           # two plain numbers
            return not abs(complex(sympy.N(g - t))) > 1e-9
    except Exception:
        pass
    return None


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
# Judge call: plain HTTP to an OpenAI-compatible /chat/completions endpoint
# (same request the openai client would send; no retries, so one timeout = one failure)
# ---------------------------------------------------------------------------

def judge_request(base_url, api_key, model, prompt, timeout, max_tokens):
    """Returns (reply_text, finish_reason). Raises on HTTP/network errors."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # thinking off: a one-word verdict doesn't need a reasoning chain. Servers that
        # don't support the option ignore it.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = data["choices"][0]
    return (choice.get("message") or {}).get("content") or "", choice.get("finish_reason")


def llm_judge_match(generated, ground_truth, problem_text, judge, judge_model):
    """True / False, or None when there's no usable verdict (call failed, timed out,
    or the reply had no YES/NO) so the record stays unjudged instead of silently wrong.
    `judge` is {"base_url", "api_key", "timeout", "max_tokens"}."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        problem=problem_text, ground_truth=ground_truth, candidate=generated)
    try:
        text, finish = judge_request(judge["base_url"], judge["api_key"], judge_model, prompt,
                                     judge["timeout"], judge["max_tokens"])
        verdict = parse_judge_verdict(text)
        if verdict is None:
            why = ("ran out of tokens before answering -- raise --judge-max-tokens"
                   if finish == "length" else "reply had no YES/NO in it")
            print(f"    (judge gave no usable verdict: {why} -- leaving this answer unjudged)")
        return verdict
    except Exception as e:
        hint = " -- try a larger --judge-timeout" if "timed out" in str(e).lower() else ""
        print(f"    (judge call failed: {e}{hint} -- leaving this answer unjudged)")
        return None


# ---------------------------------------------------------------------------
# results.json processing
# ---------------------------------------------------------------------------

def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def summarize(data, title):
    print(f"\n{title}")
    print(f"  {'adapter':26s} {'correct':>9s} {'unjudged':>9s}")
    for key, recs in data.items():
        c = sum(bool(r.get("correct")) for r in recs)
        u = sum(r.get("method") == "unjudged" for r in recs)
        print(f"  {key:26s} {c:4d}/{len(recs):<4d} {u:9d}")


def judge_one(rec, judge, judge_label):
    """Returns (verdict, method) where verdict None means 'still unjudged'."""
    mech = mechanical_match(rec["pred"], rec["gold"])
    if mech is not None:
        return mech, ("mechanical" if mech else "mechanical_mismatch")
    if judge is None:
        return None, "needs_judge"
    return llm_judge_match(rec["pred"], rec["gold"], rec["problem"], judge, judge_label), "llm_judge"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="input results.json")
    ap.add_argument("--out", default="results_rejudged.json", help="output file (default results_rejudged.json)")
    ap.add_argument("--judge-host", default="localhost")
    ap.add_argument("--judge-port", type=int, default=8081)
    ap.add_argument("--judge-model-label", default=None,
                    help="judge model name. Omit for a mechanical-only pass.")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--judge-timeout", type=float, default=None)
    ap.add_argument("--judge-max-tokens", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1, help="concurrent judge calls (default 1)")
    ap.add_argument("--checkpoint-every", type=int, default=25, help="save progress every N judged records")
    args = ap.parse_args()

    timeout = args.judge_timeout or JUDGE_TIMEOUT
    max_tokens = args.judge_max_tokens or JUDGE_MAX_TOKENS

    source = args.out if os.path.exists(args.out) else args.results
    if source == args.out:
        print(f"Resuming from existing {args.out}")
    with open(source, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        sys.exit("expected results.json to be a dict of {adapter: [records]}")

    summarize(data, "Before")

    judge = None
    if args.judge_model_label:
        base = build_base_url(args.judge_host, args.judge_port)
        judge = {"base_url": base, "timeout": timeout, "max_tokens": max_tokens,
                 "api_key": args.api_key or os.environ.get("OPENAI_API_KEY", "not-needed")}
        print(f"\nJudge: {args.judge_model_label} at {base}")
    else:
        print("\nMechanical only (no --judge-model-label given)")

    todo = [(k, i) for k, recs in data.items() for i, r in enumerate(recs)
            if r.get("method") == "unjudged" and r.get("pred") is not None]
    print(f"{len(todo)} unjudged record(s) to process")

    lock = threading.Lock()
    stats = Counter()
    by_domain = defaultdict(Counter)
    done = 0

    def apply(k, i, verdict, method):
        nonlocal done
        rec = data[k][i]
        with lock:
            if verdict is None:
                stats["still_unjudged" if method != "needs_judge" else "needs_judge"] += 1
                return
            rec["original_method"] = rec["method"]
            rec["mechanical"] = None if method == "llm_judge" else verdict
            rec["correct"] = bool(verdict)
            rec["method"] = method
            stats[method] += 1
            by_domain[rec["domain"]][("correct" if verdict else "wrong")] += 1
            done += 1
            if done % args.checkpoint_every == 0:
                save(args.out, data)
                print(f"  ...{done} judged, checkpoint saved")

    def work(item):
        k, i = item
        return k, i, *judge_one(data[k][i], judge, args.judge_model_label)

    try:
        if args.workers > 1 and judge is not None:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(work, it) for it in todo]
                for fut in as_completed(futs):
                    apply(*fut.result())
        else:
            for it in todo:
                apply(*work(it))
    except KeyboardInterrupt:
        print("\nInterrupted -- saving progress; re-run the same command to resume.")
    finally:
        save(args.out, data)

    summarize(data, "After")
    print(f"\nOutcomes: {dict(stats)}")
    if by_domain:
        print("Newly decided, by domain:")
        for dom, c in sorted(by_domain.items()):
            print(f"  {dom:14s} {c['correct']:3d} correct, {c['wrong']:3d} wrong")
    if stats["needs_judge"]:
        print(f"\n{stats['needs_judge']} record(s) need an LLM judge -- re-run with --judge-model-label.")
    if stats["still_unjudged"]:
        print(f"\n{stats['still_unjudged']} judge call(s) failed and were left unjudged -- re-run to retry them.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
