This is an experiment I did on the fine tuning of a small model to work in a harness

the nreasoning directory is the data without reasoning and the reasoning directory is the data with reasoning

Both were prepared using prepare_data.py using this command
```python prepare_data.py <directory> --out data --verified-only```

This was the domain count:
domains: {'arc': 61, 'commonsense': 57, 'deepmath': 9, 'gsm8k': 54, 'hotpot': 23, 'logic_mcq': 19, 'logiqa': 31, 'math': 20, 'mmlu': 44, 'numina_pro': 3, 'strategyqa': 36}

the model tuned is Pythia1B from this ID: `EleutherAI/pythia-1b-deduped`

nonreasoning was trained using this command
```python train_lora.py data_cot.json ./adapter_nonreasoning```

reasoning was trained using this command
```python train_lora.py data_think.json ./adapter_reasoning```

answer was trained using this command
```python train_lora.py data_answer.json ./adapter_answer```



The results were evaluated using this command
```python eval_adapters.py ./adapter_answer:24 ./adapter_reasoning:700 ./adapter_nonreasoning:700 --save results.json```
This command runs inference with each LORA on 100 questions from certain datasets. It saves the inference results, which is a seperate test suite, to results.json.
