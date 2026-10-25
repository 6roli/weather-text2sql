# %% [markdown]
# # Fine-tuning a small LLM for weather text-to-SQL
#
# QLoRA fine-tuning of **Qwen2.5-Coder-1.5B-Instruct** or **0.5B-Instruct** (both Apache 2.0) with [Unsloth](https://github.com/unslothai/unsloth),
# on a free Kaggle GPU (T4). The model learns to answer questions about Spain's weather warehouse with one DuckDB query.
#
# The notebook measures the base model first (baseline) and the fine-tuned model afterwards, with the same prompt and the
# same execution-accuracy evaluation, on two sets: the generated test split and 40 hand-written questions.
#
# **Kaggle settings:** Accelerator → GPU T4 x2 (one is used) · Internet → On. Runs in about 45 minutes.

# %%
import os
import shutil
import subprocess

if not shutil.which("nvidia-smi") or subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
    raise SystemExit("No GPU in this session: Settings → Accelerator → GPU T4 x2 (needs a phone-verified account), then run again from the top.")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Unsloth trains on one GPU

# %%
# !pip install -q unsloth duckdb==1.5.6
# !if [ -d weather-text2sql ]; then git -C weather-text2sql pull -q; else git clone -q https://github.com/6roli/weather-text2sql.git; fi

# %%
import sys

REPO = "weather-text2sql" if os.path.isdir("weather-text2sql") else ".."
sys.path.insert(0, REPO)

# %% [markdown]
# ## Settings

# %%
# 1.5B: best accuracy · 0.5B: ~2.5x smaller download for the browser demo ("unsloth/Qwen2.5-Coder-0.5B-Instruct").
# Avoid 3B: its licence is not Apache 2.0.
BASE_MODEL = "unsloth/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
OUTPUT_DIR = "outputs"
PUSH_TO_HUB = True  # uses the Kaggle secret HF_TOKEN (write access) if it exists
# one Hub repository per base model, e.g. groli/weather-text2sql-qwen2.5-coder-1.5b
HUB_REPO = "groli/weather-text2sql-" + BASE_MODEL.split("/")[-1].lower().removesuffix("-instruct")
HUB_PRIVATE = True  # made public later, when the browser demo needs to download it

# %% [markdown]
# ## Data

# %%
import json
from pathlib import Path

from text2sql.evaluate import clean_prediction, score
from text2sql.schema import messages
from text2sql.snapshot import connect


def load(name):
    return [json.loads(line) for line in (Path(REPO) / "data" / name).read_text(encoding="utf-8").splitlines() if line]


train_rows, val_rows = load("train.jsonl"), load("val.jsonl")
eval_sets = {"test": load("test.jsonl"), "handwritten": load("handwritten_test.jsonl")}
print(len(train_rows), "train ·", len(val_rows), "val ·", {k: len(v) for k, v in eval_sets.items()})
warehouse = connect(Path(REPO) / "data" / "snapshot")

# %% [markdown]
# ## Base model

# %%
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

model, tokenizer = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True)
print(f"prompt tokens: {len(tokenizer.apply_chat_template(messages(train_rows[0]['question'], train_rows[0]['today']), tokenize=True))}")

# %%
from tqdm.auto import tqdm


def predict(rows, batch_size=16, max_new_tokens=200):
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    outputs = []
    for start in tqdm(range(0, len(rows), batch_size)):
        batch = rows[start : start + batch_size]
        prompts = [tokenizer.apply_chat_template(messages(r["question"], r["today"]), tokenize=False, add_generation_prompt=True) for r in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outputs += tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return outputs


def evaluate(stage):
    results = {}
    os.makedirs("results", exist_ok=True)
    for name, rows in eval_sets.items():
        examples = [dict(r, prediction=p) for r, p in zip(rows, predict(rows))]
        results[name] = score(examples, warehouse)
        with open(f"results/{stage}_{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
        print(stage, name, results[name])
    return results


baseline = evaluate("base")

# %%
wrong = [json.loads(l) for l in open("results/base_handwritten.jsonl", encoding="utf-8")]
for e in [e for e in wrong if not e["correct"]][:5]:
    print(f"Q: {e['question']}\n   gold: {e['sql']}\n   base: {clean_prediction(e['prediction'])}\n")

# %% [markdown]
# ## QLoRA fine-tuning
#
# Only small LoRA adapters (~1% of the weights) are trained on top of the frozen 4-bit model, and the loss is computed on
# the answer (the SQL) only, not on the long schema prompt.

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=2026,
)

# %%
import dataclasses
import inspect

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only


def to_text(rows):
    texts = [tokenizer.apply_chat_template(messages(r["question"], r["today"], r["sql"]), tokenize=False) for r in rows]
    return Dataset.from_dict({"text": texts})


# argument names changed between TRL versions
config_fields = {f.name for f in dataclasses.fields(SFTConfig)}
length_arg = "max_length" if "max_length" in config_fields else "max_seq_length"
tokenizer_arg = "processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else "tokenizer"

tokenizer.padding_side = "right"
trainer = SFTTrainer(
    model=model,
    train_dataset=to_text(train_rows),
    eval_dataset=to_text(val_rows),
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=round(0.05 * EPOCHS * len(train_rows) / (BATCH_SIZE * GRAD_ACCUMULATION)),  # 5% of the steps
        weight_decay=0.01,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        report_to="none",
        seed=2026,
        **{length_arg: MAX_SEQ_LENGTH},
    ),
    **{tokenizer_arg: tokenizer},
)
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

# %%
FastLanguageModel.for_training(model)
stats = trainer.train()
print(f"{stats.metrics['train_runtime'] / 60:.1f} min, final loss {stats.metrics['train_loss']:.4f}")

# %% [markdown]
# ## Fine-tuned model

# %%
finetuned = evaluate("finetuned")

summary = {
    "base_model": BASE_MODEL,
    "settings": {"lora_rank": LORA_RANK, "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "effective_batch": BATCH_SIZE * GRAD_ACCUMULATION},
    "train_examples": len(train_rows),
    "train_minutes": round(stats.metrics["train_runtime"] / 60, 1),
    "base": baseline,
    "finetuned": finetuned,
    "loss_history": [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h],
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
for name in eval_sets:
    print(f"{name:12s} execution accuracy  base {baseline[name]['execution_accuracy']:.1%}  →  fine-tuned {finetuned[name]['execution_accuracy']:.1%}")

# %%
def ask(question, today="2026-10-25"):
    sql = clean_prediction(predict([{"question": question, "today": today}])[0])
    print(sql)
    return warehouse.sql(sql)


ask("¿Dónde hará más calor mañana?")

# %% [markdown]
# ## Save
#
# LoRA adapters (~70 MB) and a merged 16-bit model (for conversion to browser/ONNX or GGUF formats later).
# They are uploaded to the Hugging Face Hub (private) when the Kaggle secret HF_TOKEN exists; otherwise download
# results.zip and merged_16bit from the Output panel before the session ends.

# %%
model.save_pretrained("lora_adapters")
tokenizer.save_pretrained("lora_adapters")
model.save_pretrained_merged("merged_16bit", tokenizer, save_method="merged_16bit")

# !zip -r -q results.zip results

if PUSH_TO_HUB:
    # Files in /kaggle/working disappear with an interactive session; the Hub keeps them.
    from huggingface_hub import HfApi
    from kaggle_secrets import UserSecretsClient

    try:
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = None
    if token:
        api = HfApi(token=token)
        api.create_repo(HUB_REPO, private=HUB_PRIVATE, exist_ok=True)
        api.upload_folder(folder_path="merged_16bit", repo_id=HUB_REPO)
        api.upload_folder(folder_path="lora_adapters", repo_id=HUB_REPO, path_in_repo="lora_adapters")
        api.upload_folder(folder_path="results", repo_id=HUB_REPO, path_in_repo="results")
        print(f"uploaded to https://huggingface.co/{HUB_REPO} ({'private' if HUB_PRIVATE else 'public'})")
    else:
        print("No HF_TOKEN secret (Add-ons → Secrets): download results.zip and merged_16bit from the Output panel instead.")
