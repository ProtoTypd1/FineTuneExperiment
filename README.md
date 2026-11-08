# Fine-tuning a small model to work in a harness

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

### Dataset Domain counts

| Domain | Samples |
|---|---:|
| arc | 61 |
| commonsense | 57 |
| gsm8k | 54 |
| mmlu | 44 |
| strategyqa | 36 |
| logiqa | 31 |
| hotpot | 23 |
| math | 20 |
| logic_mcq | 19 |
| deepmath | 9 |
| numina_pro | 3 |
| **Total** | **357** |

## Evaluation

```
python eval_adapters.py ./adapter_answer:24 ./adapter_reasoning:700 ./adapter_nonreasoning:700 --save results.json
```

This runs inference with each LoRA on 100 questions from each of two datasets (ARC-Challenge and GSM8K, 200 questions total). The questions come from a separate test suite (the datasets' test splits), not the training data. Every generation is saved to `results.json`. The number after each adapter is its max new tokens (24 for answer-only, 700 for the reasoning adapters).

## Results

| Adapter | Accuracy | Correct | ARC | GSM8K | No `FINAL ANSWER` | Avg tokens | Hit token cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `adapter_answer` | 16.5% | 33/200 | 31/100 | 2/100 | 0 | 8 | 0/200 |
| `adapter_reasoning` | 0.0% | 0/200 | 0/100 | 0/100 | 185 | 700 | 200/200 |
| `adapter_nonreasoning` | 0.0% | 0/200 | 0/100 | 0/100 | 194 | 700 | 200/200 |
| *Most-common-answer baseline* | *15.5%* | *31/200* | *31/100* | *3/100* | | | |

The baseline always guesses the most frequent gold answer in the eval set (`B` for ARC, the most common number for GSM8K).

### Observations

- **Answer-only:** it writes `FINAL ANSWER: X` and stops, at about 8 tokens per question. Its ARC score of 31% equals the baseline: it only ever predicted `A` (66 times) and `C` (34 times), never `B` or `D`. On GSM8K it scored 2% against a 3% baseline.
- **Both reasoning adapters scored 0/200:** Every generation ran to the 700-token cap, and about 80% of them are loops. In 161/200 (reasoning) and 166/200 (nonreasoning) outputs, more than half of the 5-word sequences are repeats, for example "Let's use calculator." repeated many times, or "A ratio of 7:11 is a ratio of 7:11."
- **Content was not learned:** `adapter_reasoning` opened with a `<think>` block in 200/200 outputs, and `adapter_nonreasoning` opened with the `GIVENS:` section in 200/200.
- **Almost no final answers:** only 15 (reasoning) and 6 (nonreasoning) generations wrote a `FINAL ANSWER` line at all, and none were correct.

### Conclusion

This is by no means a conclusive test. Fine tuning should have taken place with a much larger dataset and evaluated with more diverse answers. However, on such a small scale, it appears that reasoning must either be in much larger amounts, or doesnt help at all.
