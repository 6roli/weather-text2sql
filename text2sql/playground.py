"""Local playground to inspect the fine-tuned model: where it fails, and how the browser (ONNX) version differs.

Two tabs:
- "Hand-written errors": the 40 hand-written questions with the saved predictions of the PyTorch model and of the
  browser model (ONNX q4f16), side by side with the reference query and what each query returns. No model needed.
- "Ask": type a question; both models run on this PC (CPU) and their queries are executed on the snapshot.

    pip install -e ".[playground]"
    python -m text2sql.playground

The first question downloads the models from Hugging Face (PyTorch ~3 GB, ONNX ~1.2 GB) and loads them
(PyTorch in float32 needs ~7 GB of RAM).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from functools import cache
from pathlib import Path

import duckdb

from .evaluate import Result, clean_prediction, run, same_result
from .schema import messages
from .snapshot import connect

REPO = "groli/weather-text2sql-qwen2.5-coder-1.5b"
ROOT = Path(__file__).resolve().parent.parent
HANDWRITTEN = ROOT / "data" / "handwritten_test.jsonl"
LOCAL_RESULTS = ROOT / "results" / "qwen2.5-coder-1.5b"
SNAPSHOT_TODAY = "2026-10-25"  # last day in data/snapshot; questions about "today" only make sense around it
MAX_NEW_TOKENS = 200
SHOWN_ROWS = 200


# ---------- data ----------

def _jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def _results_file(name: str) -> Path:
    """Saved predictions: the local results/ folder, or the copy uploaded next to the model."""
    local = LOCAL_RESULTS / name
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, f"results/{name}"))


@cache
def connection() -> duckdb.DuckDBPyConnection:
    return connect()


def execute(sql: str) -> tuple[Result | None, str | None]:
    if not sql:
        return None, "no query"
    try:
        return run(connection().cursor(), sql), None
    except (duckdb.Error, TimeoutError) as error:
        return None, str(error).splitlines()[0][:300]


def as_table(result: Result | None, error: str | None):
    import pandas as pd

    if result is None:
        return pd.DataFrame({"error": [error]})
    columns = [f"{name} ({i})" if result.columns.count(name) > 1 else name for i, name in enumerate(result.columns)]
    return pd.DataFrame(result.rows[:SHOWN_ROWS], columns=columns)


def sql_diff(reference: str, prediction: str) -> list[tuple[str, str | None]]:
    """Word-level diff for gr.HighlightedText: '-' only in the reference, '+' only in the prediction."""
    a, b = re.findall(r"\S+|\s+", reference), re.findall(r"\S+|\s+", prediction)
    out: list[tuple[str, str | None]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(("".join(a[i1:i2]), None))
            continue
        if i2 > i1:
            out.append(("".join(a[i1:i2]), "-"))
        if j2 > j1:
            out.append(("".join(b[j1:j2]), "+"))
    return out


# ---------- models ----------

@cache
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REPO)


def prompt(question: str, today: str) -> str:
    return tokenizer().apply_chat_template(messages(question, today), tokenize=False, add_generation_prompt=True)


@cache
def torch_model():
    import torch
    from transformers import AutoModelForCausalLM

    # float32: bfloat16/float16 matmuls are very slow on most CPUs
    return AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()


def generate_torch(question: str, today: str) -> str:
    import torch

    model, tok = torch_model(), tokenizer()
    inputs = tok(prompt(question, today), return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


@cache
def onnx_session():
    """The browser model, run with onnxruntime (same file transformers.js loads)."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    session = ort.InferenceSession(hf_hub_download(REPO, "onnx/model_q4f16.onnx"), providers=["CPUExecutionProvider"])
    return session, AutoConfig.from_pretrained(REPO)


def generate_onnx(question: str, today: str) -> str:
    """Greedy decoding with a KV cache, the same loop used to score the ONNX model in notebooks/convert_web.py."""
    import numpy as np

    session, config = onnx_session()
    tok = tokenizer()
    inputs = {i.name: i for i in session.get_inputs()}
    outputs = [o.name for o in session.get_outputs()]
    past_names = [n for n in inputs if n.startswith("past_key_values")]
    present_index = {n: outputs.index(n.replace("past_key_values", "present")) for n in past_names}
    past_dtype = np.float16 if "float16" in inputs[past_names[0]].type else np.float32
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    eos = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    eos = set(eos) | {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

    ids = tok(prompt(question, today), add_special_tokens=False)["input_ids"]
    past = {n: np.zeros((1, config.num_key_value_heads, 0, head_dim), dtype=past_dtype) for n in past_names}
    step_ids, total, new = ids, 0, []
    for _ in range(MAX_NEW_TOKENS):
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
    return tok.decode(new, skip_special_tokens=True)


# ---------- UI ----------

MARK = {True: "✅", False: "❌", None: "—"}


def build(handwritten: dict[str, dict], torch_rows: dict[str, dict], onnx_rows: dict[str, dict]):
    import gradio as gr

    def overview(filter_by: str):
        rows = []
        for key, row in handwritten.items():
            t, o = torch_rows.get(key, {}).get("correct"), onnx_rows.get(key, {}).get("correct")
            if filter_by == "Web fails" and o is not False:
                continue
            if filter_by == "PyTorch fails" and t is not False:
                continue
            if filter_by == "Models disagree" and t == o:
                continue
            rows.append([key, row["lang"], row["question"], MARK[t], MARK[o]])
        return rows, gr.update(choices=[f"{r[0]} · {r[2]}" for r in rows], value=None)

    def detail(choice: str | None):
        if not choice:
            return [None] * 10
        key = choice.split(" · ")[0]
        row = handwritten[key]
        t_sql = clean_prediction(torch_rows.get(key, {}).get("prediction", ""))
        o_sql = clean_prediction(onnx_rows.get(key, {}).get("prediction", ""))
        reference, t_res, o_res = execute(row["sql"]), execute(t_sql), execute(o_sql)
        def verdict(saved: dict, res) -> str:
            ok = saved.get("correct")
            note = f" · error: `{saved['error']}`" if saved.get("error") else ""
            if ok is None and res[0] is not None and reference[0] is not None:
                ok = same_result(reference[0], res[0])
            return f"{MARK[ok]}{note}"
        header = (f"### {key} ({row['lang']}) — today = {row['today']}\n**{row['question']}**\n\n"
                  f"PyTorch: {verdict(torch_rows.get(key, {}), t_res)} &nbsp;·&nbsp; "
                  f"Web (ONNX q4f16): {verdict(onnx_rows.get(key, {}), o_res)}")
        return (header, row["sql"], t_sql, o_sql, sql_diff(row["sql"], t_sql), sql_diff(row["sql"], o_sql),
                as_table(*reference), as_table(*t_res), as_table(*o_res), choice)

    def ask(question: str, today: str, reference_sql: str, use_torch: bool, use_onnx: bool):
        """Generator: shows each model's answer as soon as it is ready."""
        empty = as_table(None, "not run")
        state = {"status": "", "t_sql": "", "o_sql": "", "t_tab": empty, "o_tab": empty, "r_tab": empty,
                 "t_diff": None, "o_diff": None}
        def emit():
            return tuple(state[k] for k in ("status", "t_sql", "o_sql", "t_tab", "o_tab", "r_tab", "t_diff", "o_diff"))
        if not question.strip():
            state["status"] = "Write a question first."
            yield emit()
            return
        today = today.strip() or SNAPSHOT_TODAY
        reference = execute(reference_sql.strip()) if reference_sql.strip() else (None, None)
        if reference_sql.strip():
            state["r_tab"] = as_table(*reference)
        lines, results = [], {}
        for name, enabled, fn, sql_key, tab_key, diff_key in (
            ("PyTorch", use_torch, generate_torch, "t_sql", "t_tab", "t_diff"),
            ("Web (ONNX q4f16)", use_onnx, generate_onnx, "o_sql", "o_tab", "o_diff"),
        ):
            if not enabled:
                continue
            state["status"] = "\n".join(lines + [f"⏳ {name}: generating (the first time also downloads and loads it)…"])
            yield emit()
            start = time.perf_counter()
            try:
                sql = clean_prediction(fn(question, today))
            except Exception as error:  # show model/runtime failures in the UI instead of crashing the app
                lines.append(f"⚠️ {name}: {type(error).__name__}: {error}")
                continue
            seconds = time.perf_counter() - start
            res = execute(sql)
            results[name] = res[0]
            state[sql_key], state[tab_key] = sql, as_table(*res)
            if reference_sql.strip():
                state[diff_key] = sql_diff(reference_sql.strip(), sql)
            if res[1]:
                outcome = f"invalid SQL ({res[1]})"
            elif reference[0] is not None:
                outcome = "correct ✅" if same_result(reference[0], res[0]) else "wrong ❌"
            else:
                outcome = f"{len(res[0].rows)} rows"
            lines.append(f"**{name}** · {seconds:.1f} s · {outcome}")
        t, o = results.get("PyTorch"), results.get("Web (ONNX q4f16)")
        if t is not None and o is not None:
            lines.append("Both models return **the same data**." if same_result(t, o) and same_result(o, t)
                         else "The two models return **different data**.")
        state["status"] = "\n\n".join(lines)
        yield emit()

    diff_colors = {"-": "#f87171", "+": "#4ade80"}
    with gr.Blocks(title="weather-text2sql playground") as app:
        gr.Markdown("# weather-text2sql playground\nQueries run on the frozen snapshot in `data/snapshot` "
                    f"(last day {SNAPSHOT_TODAY}). In the diffs, red = only in the reference, green = only in the model.")
        with gr.Tabs() as tabs:
            with gr.Tab("Hand-written errors", id="errors"):
                t_ok = sum(bool(r.get("correct")) for r in torch_rows.values())
                o_ok = sum(bool(r.get("correct")) for r in onnx_rows.values())
                gr.Markdown(f"Saved evaluation: PyTorch **{t_ok}/{len(torch_rows)}**, "
                            f"web (ONNX q4f16) **{o_ok}/{len(onnx_rows)}** correct.")
                filter_by = gr.Radio(["Web fails", "PyTorch fails", "Models disagree", "All"], value="Web fails",
                                     label="Show")
                table = gr.Dataframe(headers=["id", "lang", "question", "PyTorch", "Web"], interactive=False, wrap=True)
                pick = gr.Dropdown(label="Open a question", choices=[])
                header = gr.Markdown()
                with gr.Row():
                    ref_sql = gr.Code(label="Reference", language="sql")
                    t_sql = gr.Code(label="PyTorch", language="sql")
                    o_sql = gr.Code(label="Web (ONNX q4f16)", language="sql")
                t_diff = gr.HighlightedText(label="Reference → PyTorch", color_map=diff_colors, combine_adjacent=True)
                o_diff = gr.HighlightedText(label="Reference → Web", color_map=diff_colors, combine_adjacent=True)
                with gr.Row():
                    ref_tab = gr.Dataframe(label="Reference result")
                    t_tab = gr.Dataframe(label="PyTorch result")
                    o_tab = gr.Dataframe(label="Web result")
                chosen = gr.State()
                rerun = gr.Button("Re-run this question with the local models →")

            with gr.Tab("Ask", id="ask"):
                with gr.Row():
                    question = gr.Textbox(label="Question (Spanish or English)", scale=4,
                                          placeholder="¿Qué capital tendrá la máxima más alta mañana?")
                    today = gr.Textbox(label="Today (UTC)", value=SNAPSHOT_TODAY, scale=1)
                reference_sql = gr.Textbox(label="Reference SQL (optional, to judge the answers)", lines=2)
                with gr.Row():
                    use_torch = gr.Checkbox(label="PyTorch (normal model)", value=True)
                    use_onnx = gr.Checkbox(label="Web model (ONNX q4f16)", value=True)
                    go = gr.Button("Ask", variant="primary")
                status = gr.Markdown()
                with gr.Row():
                    a_t_sql = gr.Code(label="PyTorch", language="sql")
                    a_o_sql = gr.Code(label="Web (ONNX q4f16)", language="sql")
                a_t_diff = gr.HighlightedText(label="Reference → PyTorch", color_map=diff_colors, combine_adjacent=True)
                a_o_diff = gr.HighlightedText(label="Reference → Web", color_map=diff_colors, combine_adjacent=True)
                with gr.Row():
                    a_r_tab = gr.Dataframe(label="Reference result")
                    a_t_tab = gr.Dataframe(label="PyTorch result")
                    a_o_tab = gr.Dataframe(label="Web result")

        filter_by.change(overview, filter_by, [table, pick])
        app.load(overview, filter_by, [table, pick])
        pick.change(detail, pick, [header, ref_sql, t_sql, o_sql, t_diff, o_diff, ref_tab, t_tab, o_tab, chosen])

        def send(choice: str | None):
            if not choice:
                return gr.update(), gr.update(), gr.update(), gr.update()
            row = handwritten[choice.split(" · ")[0]]
            return row["question"], row["today"], row["sql"], gr.Tabs(selected="ask")

        rerun.click(send, chosen, [question, today, reference_sql, tabs])
        ask_outputs = [status, a_t_sql, a_o_sql, a_t_tab, a_o_tab, a_r_tab, a_t_diff, a_o_diff]
        go.click(ask, [question, today, reference_sql, use_torch, use_onnx], ask_outputs)
        question.submit(ask, [question, today, reference_sql, use_torch, use_onnx], ask_outputs)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    handwritten = _jsonl(HANDWRITTEN)
    torch_rows = _jsonl(_results_file("finetuned_handwritten.jsonl"))
    onnx_rows = _jsonl(_results_file("onnx_q4f16_handwritten.jsonl"))
    build(handwritten, torch_rows, onnx_rows).queue().launch(server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
