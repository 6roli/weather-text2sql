# Weather text-to-SQL

Fine-tuning a small open LLM so it answers questions about Spain's weather — *"¿Dónde hará más calor mañana?"*,
*"How much rain fell in Zaragoza yesterday?"* — by writing one DuckDB query over the
[weather warehouse](https://github.com/6roli/weather-warehouse). A module of **Horizonte**.

| | |
|---|---|
| Base model | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) (Apache 2.0) |
| Method | QLoRA (4-bit base + LoRA adapters) with [Unsloth](https://github.com/unslothai/unsloth) |
| Training | Free Kaggle GPU (T4), ~45 min |
| Data | ~3,000 Spanish/English question → SQL pairs, each one checked by running it |
| Metric | Execution accuracy: does the query return the right data? |
| Cost | €0 |

## How it works

```
question families (templates.py)          40 hand-written questions
        │  29 families · Spanish + English · city aliases          │
        ▼                                                          │
generate.py ── runs every query on a frozen snapshot of the gold layer, keeps the ones with data
        │
        ├── train.jsonl (2,841)   val.jsonl (168)   test.jsonl (343)   handwritten_test.jsonl (40)
        ▼
notebooks/finetune.ipynb (Kaggle)
        1. base model → evaluate (baseline)
        2. QLoRA fine-tuning, loss on the SQL answer only
        3. fine-tuned model → evaluate, save adapters + merged model
```

**The prompt** ([schema.py](text2sql/schema.py)) is the same for training, baseline and demo: the four gold tables as
`CREATE TABLE` statements, the 52 capital ids, today's date and the question. The model answers with SQL only.

**Dates.** Relative days are written from today's date (`DATE '2026-10-25' + INTERVAL 1 DAY` for "mañana"). To stop
the model from memorising the snapshot's date, 70% of training examples have every date shifted by a random number
of days; the generator checks that the shifted example is the validated one with only its dates moved.

**Frozen data.** The live warehouse changes every hour, so [data/snapshot/](data/snapshot/) holds a 2.4 MB copy of the
gold tables (forecast issued 2026-10-25 10:00 UTC, observations 18–25 Oct). All examples are validated and scored on it.

**Evaluation** ([evaluate.py](text2sql/evaluate.py)). A prediction is correct when it returns the same data as the
reference query: row order, column names and extra columns are ignored, floats are compared to 2 decimals, and a capital
may come back as its id or its name. The **hand-written set** uses wording the templates never produce, so it measures
whether the model generalises rather than recognising templates.

## Results

Qwen2.5-Coder-1.5B-Instruct, same prompt before and after fine-tuning
([results/summary.json](results/summary.json), every prediction in [results/](results/)):

| Evaluation set | Base model | Fine-tuned | Valid SQL (base → fine-tuned) |
|---|---|---|---|
| **40 hand-written questions** (wording the templates never produce) | 7.5% | **87.5%** | 30% → 100% |
| Generated test split (343, held-out queries) | 5.5% | 100% | 24% → 100% |

Training: 2 epochs, 356 steps, 51 min on a free Kaggle T4, 18.5M trainable parameters (1.2% of the model).
Training loss fell from 0.52 to ~0.002; validation loss from 0.028 (step 50) to 0.0011, flat from step 300 — no
sign of overfitting on the generated distribution. A first run (outputs lost) scored 85% on the hand-written set:
one question can change between runs.

**What the base model gets wrong.** It knows SQL but not this warehouse: it invents columns (`temperature_max_c` in
the forecast table, `location_name` in `daily_summary`), joins three tables where one is enough (benchmark-style
`T1/T2/T3` aliases), reads tomorrow's forecast from the table of observations, and ignores "Today" in the prompt
(`NOW()`, `CURRENT_DATE`, MySQL's `DATE_SUB`).

**What the fine-tuned model still gets wrong** (5 of 40, all valid SQL):

| Question | What it did |
|---|---|
| ¿La previsión de temperatura en Madrid suele quedarse corta o pasarse? | Average forecast temperature instead of the forecast bias — colloquial wording |
| ¿Cuánto empeora la previsión de viento en Cádiz según la antelación, ayer? | One lead-time band instead of the error per lead time |
| Máximas diarias en Toledo desde el 20 de octubre | Read past days from the forecast table instead of `daily_summary` |
| Barcelona vs Valencia: ¿cuál tendrá la máxima más alta mañana? | One maximum for both cities, so the answer can't say which |
| Las 5 capitales donde peor se ha previsto la temperatura en los últimos 5 días | Ranked by `daily_summary`'s error bands instead of `forecast_vs_observed` — a defensible reading, scored wrong |

Most remaining errors are paraphrases of forecast-skill questions and past-vs-future confusion. More varied
templates would likely fix them, but the hand-written set would then no longer be unseen: a new hand-written set
is needed before claiming a better score.

## Run locally

Dataset and evaluation run on any PC (no GPU needed):

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m text2sql.generate      # rebuild data/*.jsonl from the snapshot
python -m pytest
python -m text2sql.evaluate results/finetuned_handwritten.jsonl   # score a predictions file
```

| Command | Purpose |
|---|---|
| `python -m text2sql.snapshot` | Download a new snapshot of the gold layer (changes every answer — regenerate the data after) |
| `python -m text2sql.generate` | Build train/val/test and validate the hand-written set |
| `python -m text2sql.evaluate FILE` | Execution accuracy of a predictions JSONL, with the wrong answers |
| `python notebooks/build_notebook.py` | Rebuild `finetune.ipynb` from `finetune.py` (edit the `.py`) |

## Training on Kaggle

1. Kaggle → **Create → New Notebook → File → Import notebook** → `notebooks/finetune.ipynb`.
2. Settings: **Accelerator: GPU T4 x2**, **Internet: on** (needs a phone-verified Kaggle account).
3. **Run all.** Results land in `results/` (summary + every prediction); download them from the Output tab.

## Next steps

- Browser demo: question → model (WebGPU, transformers.js/WebLLM) → SQL → DuckDB-WASM over the live bucket, no server.
- Try a smaller model (Qwen2.5-Coder-0.5B) for faster in-browser inference, and compare accuracy.

## Licence

Code: MIT. Base model: Apache 2.0 (Qwen). Weather data: Open-Meteo (CC BY 4.0) and AEMET OpenData.
