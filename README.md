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
([summary](results/qwen2.5-coder-1.5b/summary.json), every prediction in
[results/qwen2.5-coder-1.5b/](results/qwen2.5-coder-1.5b/)):

| Evaluation set | Base model | Fine-tuned | Valid SQL (base → fine-tuned) |
|---|---|---|---|
| **40 hand-written questions** (wording the templates never produce) | 7.5% | **87.5%** | 30% → 100% |
| Generated test split (343, held-out queries) | 5.5% | 100% | 24% → 100% |

Training: 2 epochs, 356 steps, 51 min on a free Kaggle T4, 18.5M trainable parameters (1.2% of the model).
Training loss fell from 0.52 to ~0.002; validation loss from 0.028 (step 50) to 0.0011, flat from step 300 — no
sign of overfitting on the generated distribution. A first run (outputs lost) scored 85% on the hand-written set:
one question can change between runs.

### 1.5B vs 0.5B

The same notebook and data with Qwen2.5-Coder-**0.5B**-Instruct
([results/qwen2.5-coder-0.5b/](results/qwen2.5-coder-0.5b/)), to see whether a model ~3x smaller — a much lighter
download for the browser — is good enough:

| | 1.5B | 0.5B |
|---|---|---|
| Hand-written questions, fine-tuned (base) | **87.5%** (7.5%) | 75.0% (0%) |
| Generated test split, fine-tuned (base) | 100% (5.5%) | 98.5% (1.2%) |
| Valid SQL on hand-written questions | 100% | 97.5% |
| Training on a Kaggle T4 | 51 min | 17 min |
| Trainable parameters | 18.5M of 1.56B (1.2%) | 8.8M of 503M (1.75%) |

The 0.5B model fails 6 questions the 1.5B answers, and its errors are the dangerous kind — valid SQL, wrong answer:
"más de 15 grados" became `temperature_c < 15`; "ayer" became two days ago; last week's rain was read from the
forecast table; the 24-hour error came from a daily band. It fixes one question the 1.5B gets wrong (daily maxima
since a past date). **The 1.5B model is the default for the demo**; the 0.5B stays available as a lighter option.

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

### Browser version (ONNX q4f16)

[convert_web.ipynb](notebooks/convert_web.ipynb) turns the merged 1.5B model into ONNX with 4-bit weights and fp16
activations (1.34 GB), the format transformers.js runs with WebGPU, and scores it on the same 40 hand-written questions
([summary](results/qwen2.5-coder-1.5b/onnx_q4f16_summary.json)):

| | PyTorch | ONNX q4f16 |
|---|---|---|
| Hand-written questions | **87.5%** | **75.0%** |
| Valid SQL | 100% | 92.5% |
| Same SQL as PyTorch | — | 26 of 40 |

The 4 bits cost 5 questions and fix none. The lost ones drop a detail rather than misread the question: a missing
`precipitation_mm > 0` or "tomorrow" filter, a missing `DISTINCT`, `>=` instead of `>` for "the last 7 days", one total
instead of one row per day. Keeping the output layer in fp16 or asymmetric 4-bit blocks may recover some of them.

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

Converting for the browser: import `notebooks/convert_web.ipynb` the same way, attach the `HF_TOKEN` secret (write),
choose `SOURCE_REPO` and run all. It uploads `onnx/model_q4f16.onnx` to the model repository only when the converted
model scores at most `MAX_DROP` below PyTorch. The training environment saves the model with transformers 5 and the
exporter needs transformers 4, so the notebook rewrites two settings first (`extra_special_tokens`, `rope_theta`);
without the `rope_theta` fix the export silently uses the wrong positions and scores 30%.

## Next steps

- Browser demo: question → model (WebGPU, transformers.js) → SQL → DuckDB-WASM over the live bucket, no server.
- Recover the accuracy lost in the 4-bit conversion (fp16 output layer, asymmetric blocks), comparing variants.

## Licence

Code: MIT. Base model: Apache 2.0 (Qwen). Weather data: Open-Meteo (CC BY 4.0) and AEMET OpenData.
