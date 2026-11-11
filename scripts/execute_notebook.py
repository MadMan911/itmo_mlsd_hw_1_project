"""Execute with this project's interpreter, without modifying user kernels."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    notebook_path = ROOT / "notebooks/01_itmo_corpus_eda.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="uniassist-kernel-") as directory:
        kernel = Path(directory) / "kernels/uniassist"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "UniAssist project environment",
                    "language": "python",
                }
            )
        )
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = directory + (os.pathsep + previous if previous else "")
        try:
            NotebookClient(
                notebook,
                timeout=180,
                kernel_name="uniassist",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
    nbformat.write(notebook, notebook_path)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    if errors:
        raise RuntimeError(errors)
    print(f"Executed {notebook_path}")


if __name__ == "__main__":
    main()
