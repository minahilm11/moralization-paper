from pathlib import Path
import argparse

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    "scripts/descriptives.ipynb",
    "scripts/moralisation_analysis.ipynb",
    "scripts/analyze_clusters.ipynb",
    "appendix/validation.ipynb",
    "appendix/examples.ipynb",
    "scripts/clustering_visuals.ipynb",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="moralization")
    args = parser.parse_args()
    output = ROOT / "results" / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    for path in NOTEBOOKS:
        print("Running " + path, flush=True)
        notebook = nbformat.read(ROOT / path, as_version=4)
        NotebookClient(
            notebook, timeout=1800, kernel_name=args.kernel,
            resources={"metadata": {"path": str((ROOT / path).parent)}},
        ).execute()
        nbformat.write(notebook, output / Path(path).name)
        print("Completed " + path, flush=True)
