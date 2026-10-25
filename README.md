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

*Pending the first training run.*

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

One-time setup:
- a Kaggle account with a **verified phone number** (Settings → Phone verification); without it the GPU and
  internet options are disabled;
- a Hugging Face **write token** (huggingface.co/settings/tokens) saved in the notebook as a secret named
  `HF_TOKEN` (**Add-ons → Secrets**). The last cell then uploads the model, the LoRA adapters and `results/` to
  `groli/weather-text2sql-qwen2.5-coder-1.5b` (private until the demo needs it). Without it, download
  `results.zip` and `merged_16bit/` from the Output panel — files in an interactive session are deleted when it ends.

1. **Create → New Notebook → File → Import Notebook**, and paste
   `https://raw.githubusercontent.com/6roli/weather-text2sql/main/notebooks/finetune.ipynb`.
2. Session options: **Accelerator → GPU T4 x2** (one is used) and **Internet → On**.
3. Run the first code cell: it stops with a clear message if the session has no GPU.
4. **Save Version → Save & Run All (Commit)**. It runs in the background (~45–60 min); the browser can be closed.
   Stop the interactive session so it doesn't use GPU quota (~30 h/week free) in parallel.
5. Open the finished version → **Output** → download `results/` (summary + every prediction) into this repository.

| Problem | Fix |
|---|---|
| `Unsloth cannot find any torch accelerator? You need a GPU.` | The GPU is not enabled (step 2); changing it restarts the session, run again from the top |
| `No module named 'text2sql'` | Internet was off, so the repository was not cloned |
| `CUDA out of memory` | `BATCH_SIZE = 4`, `GRAD_ACCUMULATION = 4` in the Settings cell |
| Red `FutureWarning` / "unauthenticated requests to the HF Hub" | Harmless warnings |

Kaggle is used instead of Google Colab because a committed run continues with the browser closed, keeps its outputs,
and the weekly GPU quota is fixed.

## Converting for the browser

[notebooks/convert_web.ipynb](notebooks/convert_web.ipynb) turns the merged model on the Hub into ONNX with 4-bit
weights and fp16 activations (`q4f16`), the format transformers.js runs with WebGPU, and uploads it to the same
repository under `onnx/`. Import it in Kaggle like the training notebook (same `HF_TOKEN` secret, GPU on), set
`SOURCE_REPO` to the model to convert and **Save & Run All** (~20–30 min).

- The exporter needs an older `transformers` than Kaggle ships, so it runs in its own Python 3.11 environment.
- Recipe: 4-bit matrix multiplications first (the fp32 export is too large to convert in one piece), then fp16 with
  ONNX Runtime's converter (`onnxconverter_common` leaves Qwen's RMSNorm with mixed fp32/fp16 types).
- Before uploading, the converted model answers the 40 hand-written questions again with ONNX Runtime and is scored
  with the same evaluation, so accuracy lost by the compression is measured, not assumed
  (`results/onnx_q4f16_summary.json`).
- Size: 475 MB for the 0.5B model (tested end to end); the 1.5B model is about 2.5x larger.

## Next steps

- Browser demo: question → model (WebGPU, transformers.js/WebLLM) → SQL → DuckDB-WASM over the live bucket, no server.
- Try a smaller model (Qwen2.5-Coder-0.5B) for faster in-browser inference, and compare accuracy.

## Licence

Code: MIT. Base model: Apache 2.0 (Qwen). Weather data: Open-Meteo (CC BY 4.0) and AEMET OpenData.
