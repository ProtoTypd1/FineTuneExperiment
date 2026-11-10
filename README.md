# Fine-tuning a small model

An experiment on LoRA fine-tuning of a small model (Pythia-1B, `EleutherAI/pythia-1b-deduped`) on agentic trajectories, comparing training targets with and without reasoning.

All data was prepared using this command, training on only correct answers.
```
python prepare_data.py <directory> --out data --verified-only
```

## Training setup

| Adapter | Training data | Training target | Train command |
|---|---|---|---|
| `adapter_answer` | `data_answer.json` | `FINAL ANSWER: X` only | `python train_lora.py data_answer.json ./adapter_answer` |
| `adapter_nonreasoning` | `data_cot.json` | Visible written reasoning, then `FINAL ANSWER: X` | `python train_lora.py data_cot.json ./adapter_nonreasoning` |
| `adapter_reasoning` | `data_think.json` | `<think>` reasoning blocks plus the visible reasoning, then `FINAL ANSWER: X` | `python train_lora.py data_think.json ./adapter_reasoning` |

### Dataset domains

The same 11 domains are used for training and for evaluation.

| Domain | Source dataset | Task | Train samples | Eval questions |
|---|---|---|---:|---:|
| arc | `allenai/ai2_arc` (Challenge) | Science reasoning, multiple choice | 61 | 100 |
| commonsense | `tau/commonsense_qa` | Commonsense reasoning, multiple choice | 57 | 100 |
| gsm8k | `openai/gsm8k` | Grade-school word problems | 54 | 100 |
| mmlu | `cais/mmlu` | Broad knowledge, multiple choice | 44 | 100 |
| strategyqa | `ChilleD/StrategyQA` | Multi-step yes/no reasoning | 36 | 100 |
| logiqa | `lucasmccabe/logiqa` | Logical reasoning reading comprehension | 31 | 100 |
| hotpot | `hotpotqa/hotpot_qa` | Multi-hop question answering | 23 | 100 |
| math | `AI-MO/NuminaMath-CoT` | Competition-style math | 20 | 100 |
| logic_mcq | `hivaze/LOGIC-701` (en) | Logic puzzles, multiple choice | 19 | 100 |
| deepmath | `zwhe99/DeepMath-103K` | Hard math, answer match | 9 | 100 |
| numina_pro | `Azure99/NuminaMath-1.5-Pro` | Hard-flagged math, answer match | 3 | 100 |
| **Total** | | | **357** | **1,100** |

## Evaluation

```
python eval_adapters.py ./adapter_answer:24 ./adapter_reasoning:700 ./adapter_nonreasoning:700 --save results.json
```

Each adapter is run on 100 questions from each of the 11 domains (1,100 questions per adapter). Every generation is saved to `results.json`. The number after each adapter is its max new tokens (24 for answer-only, 700 for the reasoning adapters).

> TODO: note any dataset-selection flags used for the 11-domain run, and confirm that the evaluation questions in every domain come from held-out splits rather than the training data.

### Scoring

1. **Mechanical match first:** exact, numeric, and LaTeX-aware comparison of the predicted answer against the gold answer. This settles most records.
2. **Manual review for the rest:** 256 of the 3,300 answers (7.8%) could not be settled mechanically (108 hotpot, 49 numina_pro, 42 math, 36 deepmath, 21 others). Each was reviewed individually by Claude. Result: 5 correct, 2 borderline (counted as incorrect), 249 incorrect. Do keep in mind: these are AI reviews, not human ones.

Files: `results.json` (raw generations), `results_final.json` (Claude Reviewed file).

## Results

| Adapter | Accuracy | Correct | No `FINAL ANSWER` | Avg tokens | Looped | Hit token cap |
|---|---:|---:|---:|---:|---:|---:|
| `adapter_answer` | 17.4% | 191/1100 | 0 | 9 | 0 | 13/1100 |
| `adapter_reasoning` | 2.4% | 26/1100 | 998 | 282 | 911 | 90/1100 |
| `adapter_nonreasoning` | 0.1% | 1/1100 | 1083 | 343 | 969 | 115/1100 |
| *Most-common-answer baseline* | *20.4%* | *224/1100* | | | | |

Looped generations are generations that generate the same tokens over and over again. These are generally detected and cut short by the inference script instead of generating until the token limit.
The baseline always guesses the most frequent gold answer for that domain in the eval set.

### Correct answers by domain (out of 100)

| Domain | `adapter_answer` | `adapter_reasoning` | `adapter_nonreasoning` | Baseline |
|---|---:|---:|---:|---:|
| arc | 34 | 0 | 0 | 31 |
| commonsense | 15 | 2 | 0 | 22 |
| deepmath | 11 | 1 | 0 | 17 |
| gsm8k | 2 | 0 | 0 | 3 |
| hotpot | 14 | 1 | 0 | 4 |
| logic_mcq | 15 | 1 | 0 | 26 |
| logiqa | 24 | 0 | 0 | 32 |
| math | 8 | 0 | 0 | 6 |
| mmlu | 19 | 2 | 0 | 30 |
| numina_pro | 1 | 0 | 0 | 2 |
| strategyqa | 48 | 19 | 1 | 51 |
| **Total** | **191** | **26** | **1** | **224** |

### Observations

- **No adapter beat the most-common-answer baseline overall** (20.4%). `adapter_answer` came closest at 17.4%. It was above baseline in only three domains: arc (34 vs 31), math (8 vs 6) and hotpot (14 vs 4).
- **Answer-only:** it wrote `FINAL ANSWER: X` and stopped on all 1,100 questions, at about 9 tokens each, but its predictions collapse onto one or two options. On ARC it only ever predicted `A` (70) or `C` (30). The same pattern holds for mmlu (`A` 76, `C` 24), logiqa (`A` 69, `C` 30, one `B`), commonsense (`A` 69 of 100) and logic_mcq (`A` 68 of 100). On strategyqa it said "no" 99 times. On open-ended math it often returned a stray letter (`N`, `X`, `A`) instead of an answer.
- **Both reasoning adapters almost never reached an answer.** `adapter_reasoning` opened with a `<think>` block on 1,100/1,100 questions and `adapter_nonreasoning` opened with `GIVENS:` on 1,100/1,100, but only 102 (reasoning) and 17 (nonreasoning) generations wrote a `FINAL ANSWER` line at all.
- **Looping:** 911/1100 (reasoning) and 969/1100 (nonreasoning) generations were flagged as loops, typically repeating a phrase over and over. Of the generations that did not loop, 115 nonreasoning generations ran to the 700-token cap instead.
- **When the reasoning adapter finished, it was right about a quarter of the time:** 26 of its 102 generations with a `FINAL ANSWER` line were correct, and 19 of those 26 were strategyqa.
- **Content was not learned:** both adapters reproduced the format of the training data (`<think>` blocks, the `GIVENS:` section) without producing working reasoning.

### Conclusion

This is by no means a conclusive test. Fine tuning should have taken place with a much larger dataset. However, on such a small scale, it appears that reasoning must either be in much larger amounts, or doesn't help at all. Across 1,100 questions in 11 domains, none of the three adapters beat simply guessing the most common answer.
