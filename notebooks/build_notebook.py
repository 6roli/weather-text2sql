"""Turns notebooks/finetune.py (cells separated by `# %%`) into notebooks/finetune.ipynb for Kaggle.

The .py file is the one to edit (readable diffs); run `python notebooks/build_notebook.py` afterwards.
Lines like `# !pip install ...` become real `!pip install ...` shell lines in the notebook.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cells(source: str):
    for block in source.split("# %%")[1:]:
        header, _, body = block.partition("\n")
        lines = body.strip("\n").splitlines()
        if header.strip() == "[markdown]":
            yield {"cell_type": "markdown", "metadata": {}, "source": "\n".join(line[2:] if line.startswith("# ") else line.removeprefix("#") for line in lines)}
        else:
            code = "\n".join(line[2:] if line.startswith("# !") else line for line in lines)
            yield {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code}


def main() -> None:
    notebook = {
        "cells": list(cells((HERE / "finetune.py").read_text(encoding="utf-8"))),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / "finetune.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
