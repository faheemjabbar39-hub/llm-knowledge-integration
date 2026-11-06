# Knowledge-Integration Strategies for LLMs in Asset Management

A controlled comparison of four ways to give a large language model financial knowledge:

| Strategy | What it does | Status |
|---|---|---|
| **Base** | Model used as-is, no extra knowledge | ✅ ready (stage 2) |
| **QLoRA** | Small adapters trained on finance Q&A, 4-bit model | ✅ script ready (stage 3) |
| **RAG** | Searches a document index for each question | ⏳ stage 4 |
| **CAG** | Documents preloaded, KV cache reused | ⏳ stage 5 |

Models: Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, Mistral-7B-Instruct-v0.3.
4 strategies × 3 models = **12 conditions**, all scored on the same test questions.

---

## How the data is used: training vs testing

Every dataset is split into a **train** file and a **test** file that never overlap.

| Dataset | Train (used only by QLoRA) | Test (used by all 12 conditions) |
|---|---|---|
| HotpotQA | 1,000 from the official *train* split | 500 from the official *validation* split |
| Natural Questions | 1,000 from *train* | 500 from *validation* |
| MMLU finance subjects | *dev* + *validation* questions | 500 from *test* |
| TruthfulQA | **none** (never trained on) | 500 (hallucination probe) |
| Domain (SEC 10-K + ESMA) | Q&A from ~70% of the **documents** | ~200 Q&A from the other ~30% of documents |

The domain set is split **by document**, not by question, so no passage that produced
a test question is ever seen during training.

---

## Step-by-step setup

### Step 1: Create accounts (one-time, ~15 min)
1. **GitHub**: to clone this repo.
2. **Hugging Face** (huggingface.co): to download the models.
   - Open [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and
     [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) → **request access**
     (approval usually takes minutes to a few hours).
   - Open [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) → accept the terms.
   - Settings → Access Tokens → create a **Read** token. Keep it private.
3. **Google Colab** (colab.research.google.com) or **Kaggle** (kaggle.com): free NVIDIA GPU for the runs.

### Step 2: Install software on your laptop (for writing and testing code)

**Windows**
1. Python 3.11 from python.org; tick **"Add Python to PATH"** during install.
2. Git from git-scm.com (default options).
3. VS Code from code.visualstudio.com, plus its **Python** and **Jupyter** extensions.

**Mac**
1. Install Homebrew (brew.sh), then: `brew install python@3.11 git`
2. VS Code + Python and Jupyter extensions.

### Step 3: Get the project and install packages
```bash
git clone https://github.com/<your-account>/<repo-name>.git
cd <repo-name>
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
hf auth login            # paste your Hugging Face token (older versions: huggingface-cli login)
python -m pytest -q      # should print "6 passed"
```

### Step 4: Run the baseline on a GPU (Colab/Kaggle)
Open `notebooks/run_baseline_colab.ipynb` in Colab → *Runtime → Change runtime type → T4 GPU* → run the cells top to bottom.
It does the same as these commands:
```bash
python scripts/01_prepare_benchmarks.py                                  # download + split the benchmarks
python scripts/05_run_eval.py --model llama8b --strategy base --limit 20 # 2-minute smoke test
python scripts/05_run_eval.py --model llama8b --strategy base            # full baseline (~1 h on a T4)
python scripts/05_run_eval.py --model mistral7b --strategy base
python scripts/06_analyze.py --gpu-usd-per-hour 0.35                     # summary table
```

> **Mac users:** 4-bit loading needs an NVIDIA GPU. On a Mac set `load_in_4bit: false` in
> `configs/experiment.yaml` (runs in float16, ~16 GB for the 8B model). Use Mac runs for
> development only and never mix them with GPU numbers in the final analysis, because
> latency and memory are not comparable across hardware.

### Step 5: Build the domain (finance) dataset
```bash
# 1. put your name + email in configs/experiment.yaml -> domain.sec_user_agent (SEC requirement)
python scripts/02_download_sec_filings.py      # 10-Ks of 10 asset managers
# 2. download ESMA Q&A PDFs by hand into data/raw/esma/
python scripts/03_build_domain_corpus.py       # clean, chunk, split documents train/test
python scripts/04_generate_domain_qa.py        # generate + verify Q&A (needs GPU)
```
Then open `data/processed/domain_test.jsonl` and check ~30 random items by hand.

### Step 6: QLoRA training (stage 3)
```bash
python scripts/07_train_qlora.py --model llama8b --rank 16 --seed 42
python scripts/05_run_eval.py --model llama8b --strategy qlora --adapter adapters/llama8b_r16_s42 --tag r16
```

---

## Project layout
```
configs/experiment.yaml   all settings in one place (seed, models, splits, decoding)
src/kis/                  the evaluation harness
  models.py               model layer: identical loading (4-bit NF4) and timed generation
  strategies.py           knowledge layer: the only part that changes between conditions
  prompts.py              one prompt template for every condition
  metrics.py              exact match, multiple-choice scoring, hallucination flag
  stats.py                Cochran's Q → McNemar, Friedman → Wilcoxon, Holm correction
  topsis.py               TOPSIS decision model + weight sensitivity check
scripts/01-07             numbered pipeline steps
results/raw/              one CSV per condition, one row per question
```

## Output columns (results/raw/*.csv)
`item_id, dataset, model, strategy, prediction, gold, correct (0/1), contains (0/1),
hallucinated (TruthfulQA only), latency_ms, prompt_tokens, output_tokens, device, quant`

## Notes for the write-up
- All strategies load the model in 4-bit NF4, so quantisation is held constant and not confounded with QLoRA.
- Greedy decoding (temperature 0), same prompt, randomised item order with a fixed seed.
- Domain Q&A is generated with Qwen2.5-7B-Instruct, a model **not** under test, to avoid bias.
- Still to add: BERTScore and LLM-judge scoring, RAG (stage 4), CAG (stage 5).
