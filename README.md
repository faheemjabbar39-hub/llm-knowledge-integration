# Knowledge-Integration Strategies for LLMs on Financial Reports

A controlled comparison of four ways to give a language model the knowledge it needs to
answer questions about company financial reports.

| Strategy | What the model gets | Simple picture |
|---|---|---|
| **Base** | The question only | Answers from memory |
| **RAG** | The question + the 3 report extracts found by search | Open-book exam: looks up the page |
| **CAG** | The question + a whole knowledge block, preloaded once (KV cache) | All notes already open on the desk |
| **QLoRA** | A fine-tuned model + the same extracts as RAG | Studied the skill before the exam |

Three open models of the same size class: **Llama-3.1-8B**, **Mistral-7B** and **Qwen2.5-7B**.
4 strategies × 3 models = **12 conditions**, all answering the same 504 test questions,
all loaded in 4-bit on the same GPU type.

## How to run it

Open **`notebooks/main_study.ipynb`** in Google Colab (File → Open notebook → GitHub, tick
*Include private repos*), set the runtime to **T4 GPU**, and run the cells from top to bottom.
The notebook follows the steps below.

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_prepare_tatqa.py` | Downloads TAT-QA, builds train / dev / test files and the knowledge blocks |
| 2 | `scripts/02_check_setup.py` | Checks that search finds the right extract and that every block fits each model |
| 3 | `scripts/03_run_eval.py` | Runs a model under Base, RAG, CAG (and QLoRA once trained) |
| 4 | `scripts/04_train_qlora.py` | Trains the QLoRA adapter on the train questions |
| 5 | `scripts/05_analyze.py` | Summary tables, significance tests, TOPSIS |
| 6 | `scripts/06_plot_results.py` | Charts |

Every model and strategy writes its own file (`results/raw/<model>__<strategy>.csv`), and
every run continues where it stopped, so nothing is overwritten or repeated.

## The data: TAT-QA

[TAT-QA](https://github.com/NExTplusplus/TAT-QA) (Zhu et al., ACL 2021; licence CC BY 4.0)
has 16,552 questions over 2,757 extracts from real financial reports. Each extract is a
table plus its paragraphs, with about 6 questions written and double-checked by people with
a finance background. Answers are words or values copied from the report, or calculations.

The official splits are kept, so no report extract is shared between them:

| Split | Used for | Size in this study |
|---|---|---|
| train | QLoRA training only | 200 extracts, 1,202 questions |
| dev | Checking search and block sizes (Step 2) | 56 extracts, 336 questions |
| test | All final results | 84 extracts, 504 questions |

The 84 test extracts are put into 3 **blocks** of 28. A block is one small knowledge base:
RAG searches inside it and CAG preloads all of it, so both strategies see exactly the same
knowledge and differ only in how it reaches the model. The sample is drawn with a fixed
seed (42); the resulting files are committed in `data/processed/`.

## What is measured

| Measure | Meaning |
|---|---|
| `accuracy` | Strict: the right value **and** the right scale (thousand, million, %), following the official TAT-QA rule |
| `accuracy_lenient` | The right value, even if the scale or extra words differ |
| `f1` | Word overlap for text answers |
| `hallucination_rate` | The model gave an answer and it was wrong |
| `abstain_rate` | The model said "I don't know" (counted separately from wrong answers) |
| `retrieval_hit` | RAG and QLoRA: the right extract was among those found |
| `latency_p50/p95/p99_ms` | Response time, including search time |
| `peak_gpu_gb`, `cost_per_1k_usd`, `setup_s` | GPU memory, running cost, and one-off preparation time |

Statistics: Cochran's Q then McNemar for right/wrong outcomes, Friedman then Wilcoxon for
times and scores, all Holm-corrected; TOPSIS ranks the strategies for two deployment profiles.

## Design notes for the write-up

- **Why QLoRA is tested with documents.** No model can know an unseen company's figures from
  memory, so fine-tuning here teaches the *skill* (reading financial tables, calculating,
  answering in the right format), not the facts. An extra run without documents
  (`--no-docs`) shows what fine-tuning alone achieves.
- **Why blocks.** The whole test knowledge base does not fit in a 7–8B model's memory on a
  16 GB GPU, which is itself a real limit of CAG. Blocks keep RAG and CAG on equal knowledge.
- **Questions that do not name the company.** Many TAT-QA questions are short ("What was
  revenue in 2019?"). Step 2 measures how often search still finds the right extract.
- **Controls.** Same questions, same prompt, greedy decoding, 4-bit NF4 for every strategy,
  fixed seed, same GPU type. Only the strategy and the model change.

## Project layout

```
configs/experiment.yaml   all settings in one place
notebooks/main_study.ipynb the one notebook to run
scripts/01 ... 06         the pipeline steps
src/kis/                  the harness: models, prompts, retrieval, tatqa (marking), qlora, stats, topsis
data/processed/           the TAT-QA questions and knowledge blocks used in the study
results/raw/              one CSV per model and strategy, one row per question
adapters/                 trained QLoRA adapters
tests/                    checks that run without a GPU (python -m pytest -q)
```

## Pilot study (archived)

Before the main study, Llama-3.1-8B and Mistral-7B baselines were run on general benchmarks
(HotpotQA, Natural Questions, MMLU finance subjects, TruthfulQA). Those results are kept in
`results/pilot/`, with the notebooks in `notebooks/pilot/` and the scripts in
`scripts/pilot/`. The pilot validated the harness and showed that general benchmarks do not
fit a finance question, which led to the TAT-QA design. The pilot code as it was run is at
commit `3410f0b`.
