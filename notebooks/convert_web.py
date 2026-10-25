# %% [markdown]
# # Converting the fine-tuned model for the browser
#
# Turns the merged model on the Hugging Face Hub into **ONNX with 4-bit weights and fp16 activations** (`q4f16`), the
# format [transformers.js](https://huggingface.co/docs/transformers.js) runs with WebGPU, and uploads it to the same
# repository under `onnx/`. Before uploading, the converted model answers the 40 hand-written questions again and is
# scored with the same execution-accuracy evaluation, so any accuracy lost by the 4-bit compression is measured.
#
# **Kaggle settings:** Accelerator → GPU T4 x2 (used only for the check) · Internet → On ·
# Add-ons → Secrets → `HF_TOKEN` (write) attached. Runs in about 20–30 minutes.

# %%
import os
import shutil
import subprocess

if not shutil.which("nvidia-smi") or subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
    raise SystemExit("No GPU in this session: Settings → Accelerator → GPU T4 x2, then run again from the top.")

# %% [markdown]
# ## Settings

# %%
SOURCE_REPO = "groli/weather-text2sql-qwen2.5-coder-1.5b"  # or groli/weather-text2sql-qwen2.5-coder-0.5b
UPLOAD = True  # upload onnx/ and the check's results to SOURCE_REPO

# %% [markdown]
# ## A separate Python environment
#
# The ONNX exporter supports older `transformers` versions than the one Kaggle ships (which Unsloth needs), so the
# conversion runs in its own Python 3.11 environment and doesn't touch the notebook's packages.

# %%
# !pip install -q uv
# !uv venv -q -p 3.11 /tmp/onnx-env
# !uv pip install -q -p /tmp/onnx-env "optimum-onnx[onnxruntime-gpu]" onnx onnx-ir accelerate "huggingface_hub[hf_xet]" duckdb==1.5.6 || uv pip install -q -p /tmp/onnx-env "optimum[exporters,onnxruntime-gpu]" onnx onnx-ir accelerate "huggingface_hub[hf_xet]" duckdb==1.5.6
# !if [ -d weather-text2sql ]; then git -C weather-text2sql pull -q; else git clone -q https://github.com/6roli/weather-text2sql.git; fi

# %%
from kaggle_secrets import UserSecretsClient

env = dict(os.environ, HF_TOKEN=UserSecretsClient().get_secret("HF_TOKEN"), SOURCE_REPO=SOURCE_REPO, UPLOAD=str(UPLOAD))
PYTHON = "/tmp/onnx-env/bin/python"


def run(script: str) -> None:
    """Runs a step in the ONNX environment, streaming its output."""
    subprocess.run([PYTHON, "-u", "-c", script], env=env, check=True)


run("import optimum, onnxruntime, transformers; print('transformers', transformers.__version__, '· onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())")

# %% [markdown]
# ## 1. Download the merged model and export it to ONNX
#
# Only the merged weights and tokenizer are downloaded (not the LoRA adapters or results). The export writes the
# model with its key/value cache, which makes generating each new token fast.

# %%
run(r'''
import os
from huggingface_hub import snapshot_download
from optimum.exporters.onnx import main_export

src = snapshot_download(os.environ["SOURCE_REPO"], token=os.environ["HF_TOKEN"], local_dir="model_src",
                        ignore_patterns=["lora_adapters/*", "results/*", "onnx/*"])
# no_post_process: the tied-weights clean-up serialises the whole fp32 model, which fails above 2 GB (1.5B is ~6 GB);
# it isn't needed, the duplicated output matrix is compressed to 4 bits in the next step
main_export(src, output="onnx_fp32", task="text-generation-with-past", device="cpu", no_post_process=True)
print(sorted(os.listdir("onnx_fp32")))
''')

# %% [markdown]
# ## 2. Compress to `q4f16`
#
# The large matrix multiplications go to 4-bit blocks first (the fp32 export is ~4x the model size and must shrink
# before anything is serialised: ONNX files are limited to 2 GB without external data), then every remaining weight
# and activation goes to fp16. The embedding table stays in fp16.

# %%
run(r'''
import os
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

try:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as Quantizer
except ImportError:
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer as Quantizer

quantizer = Quantizer("onnx_fp32/model.onnx", block_size=32, is_symmetric=True)
quantizer.process()
os.makedirs("onnx_q4", exist_ok=True)
quantizer.model.save_model_to_file("onnx_q4/model.onnx", use_external_data_format=True)

model = onnx.load("onnx_q4/model.onnx", load_external_data=True)
# ONNX Runtime's converter keeps Qwen's RMSNorm consistent (onnxconverter_common leaves fp32/fp16 mixed there)
quantized = convert_float_to_float16(model, keep_io_types=False, disable_shape_infer=True)
os.makedirs("web/onnx", exist_ok=True)
size_gb = quantized.ByteSize() / 1e9
print(f"q4f16 model: {size_gb:.2f} GB")
if size_gb < 1.9:  # one file when it fits in protobuf's 2 GB limit
    onnx.save(quantized, "web/onnx/model_q4f16.onnx")
else:
    onnx.save(quantized, "web/onnx/model_q4f16.onnx", save_as_external_data=True, location="model_q4f16.onnx_data")
for name in sorted(os.listdir("web/onnx")):
    print(f"{name}: {os.path.getsize(os.path.join('web/onnx', name)) / 1e6:.0f} MB")
''')

# %% [markdown]
# ## 3. Check: the converted model answers the hand-written questions again
#
# Greedy generation with ONNX Runtime on the GPU, the same prompt as in training, and the same evaluation. The result
# is compared with the PyTorch model's answers (`results/finetuned_handwritten.jsonl` in the Hub repository).

# %%
run(r'''
import json, os, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, "weather-text2sql")
from text2sql.evaluate import clean_prediction, score
from text2sql.schema import messages
from text2sql.snapshot import connect

config = AutoConfig.from_pretrained("model_src")
tokenizer = AutoTokenizer.from_pretrained("model_src")
session = ort.InferenceSession("web/onnx/model_q4f16.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = {i.name: i for i in session.get_inputs()}
outputs = [o.name for o in session.get_outputs()]
past_names = [n for n in inputs if n.startswith("past_key_values")]
present_index = {n: outputs.index(n.replace("past_key_values", "present")) for n in past_names}
past_dtype = np.float16 if "float16" in inputs[past_names[0]].type else np.float32
kv_heads = config.num_key_value_heads
head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
eos = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
eos = set(eos) | {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")}


def generate(prompt, max_new_tokens=200):
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    past = {n: np.zeros((1, kv_heads, 0, head_dim), dtype=past_dtype) for n in past_names}
    step_ids, total, new = ids, 0, []
    for _ in range(max_new_tokens):
        feeds = {"input_ids": np.array([step_ids], dtype=np.int64),
                 "attention_mask": np.ones((1, total + len(step_ids)), dtype=np.int64), **past}
        if "position_ids" in inputs:
            feeds["position_ids"] = np.arange(total, total + len(step_ids), dtype=np.int64)[None]
        result = session.run(None, feeds)
        token = int(result[0][0, -1].argmax())
        total += len(step_ids)
        if token in eos:
            break
        new.append(token)
        past = {n: result[i] for n, i in present_index.items()}
        step_ids = [token]
    return tokenizer.decode(new, skip_special_tokens=True)


rows = [json.loads(l) for l in open("weather-text2sql/data/handwritten_test.jsonl", encoding="utf-8") if l.strip()]
examples = []
for r in rows:
    prompt = tokenizer.apply_chat_template(messages(r["question"], r["today"]), tokenize=False, add_generation_prompt=True)
    examples.append(dict(r, prediction=generate(prompt)))
summary = score(examples, connect(Path("weather-text2sql/data/snapshot")))

torch_file = hf_hub_download(os.environ["SOURCE_REPO"], "results/finetuned_handwritten.jsonl", token=os.environ["HF_TOKEN"])
torch_rows = {json.loads(l)["id"]: json.loads(l) for l in open(torch_file, encoding="utf-8")}
same = sum(clean_prediction(e["prediction"]) == clean_prediction(torch_rows[e["id"]]["prediction"]) for e in examples)
summary.update(pytorch_accuracy=round(sum(t["correct"] for t in torch_rows.values()) / len(torch_rows), 4),
               identical_to_pytorch=f"{same}/{len(examples)}")
os.makedirs("results", exist_ok=True)
with open("results/onnx_q4f16_handwritten.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
json.dump(summary, open("results/onnx_q4f16_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
for e in [e for e in examples if not e["correct"]]:
    print(f"\nQ: {e['question']}\n   onnx: {clean_prediction(e['prediction'])}")
''')

# %% [markdown]
# ## 4. Upload
#
# `onnx/model_q4f16.onnx` goes next to the merged model's `config.json` and tokenizer files, the layout transformers.js
# expects: `pipeline("text-generation", "<repo>", { dtype: "q4f16", device: "webgpu" })`. The repository stays as it
# is (private or public); the browser demo needs it public.

# %%
run(r'''
import os
from huggingface_hub import HfApi

if os.environ["UPLOAD"] == "True":
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(folder_path="web/onnx", repo_id=os.environ["SOURCE_REPO"], path_in_repo="onnx")
    api.upload_folder(folder_path="results", repo_id=os.environ["SOURCE_REPO"], path_in_repo="results")
    print(f"uploaded to https://huggingface.co/{os.environ['SOURCE_REPO']}/tree/main/onnx")
''')

# %%
# !zip -r -q results_onnx.zip results
