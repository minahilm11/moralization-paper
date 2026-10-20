import argparse
from contextlib import chdir
import json
import os
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION = ROOT / "scripts" / "classification"
with patch.object(sys, "path", [str(CLASSIFICATION), *sys.path]):
    batch = runpy.run_path(str(CLASSIFICATION / "run_noshot_batches.py"))
notebook = nbformat.read(ROOT / "scripts/topic_classification.ipynb", as_version=4)


class BatchHandoffTests(unittest.TestCase):
    def test_batch_output_to_policy_topics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "scripts").mkdir()
            source = root / "data" / "speeches.parquet"
            pd.DataFrame(
                {"text": ["Speech nineteen", "Speech seven", "Speech forty-two"]},
                index=[19, 7, 42],
            ).to_parquet(source)
            args = argparse.Namespace(
                input_path=source, output_dir=root / "outputs" / "noshot_4omini_batches",
                start=0, stop=3, text_column="text", overwrite=False,
                model="gpt-4o-mini", max_tokens=5000, temperature=0.0, seed=42,
            )
            batch["prepare_batch"](args)
            chunk = args.output_dir / "batch_0_3"
            (chunk / "batch_info.json").write_text(json.dumps({"start": 0, "stop": 3}))
            results = [
                {
                    "custom_id": "row_" + str(row_idx),
                    "response": {"body": {"choices": [{"message": {
                        "content": "<thinking>Reasoning</thinking><answer>" + str(rating) + "</answer>"
                    }}]}},
                }
                for row_idx, rating in [(42, 5), (19, 3), (7, 1)]
            ]
            response = SimpleNamespace(text="\n".join(json.dumps(row) for row in results))
            client = SimpleNamespace(files=SimpleNamespace(content=lambda _: response))
            status = SimpleNamespace(status="completed", output_file_id="output", error_file_id=None)
            with patch.dict(batch["download_batch"].__globals__, {
                "show_status": lambda _: status, "openai_client": lambda: client,
            }):
                batch["download_batch"](args)
            batch["merge_batches"](args)

            scope = {}
            with chdir(root / "scripts"), patch.dict(os.environ):
                os.environ.pop("MORALIZATION_INPUT", None)
                for cell in notebook.cells[:3]:
                    exec(cell.source, scope)
                self.assertEqual(scope["df"].custom_id.tolist(), ["row_7", "row_19", "row_42"])
                self.assertEqual(scope["df"].new_rating.tolist(), [1, 3, 5])
                self.assertEqual(scope["df"].text.tolist(), ["Speech seven", "Speech nineteen", "Speech forty-two"])
                scope["predictions"] = {"row_42": "Health", "row_7": "Education", "row_19": "Economy"}
                for cell in notebook.cells[-2:]:
                    exec(cell.source, scope)
            topics = pd.read_parquet(root / "data" / "policy_topics.parquet")
            self.assertEqual(topics.original_row_idx.tolist(), [7, 19, 42])
            self.assertEqual(topics.topic_label.tolist(), ["Education", "Economy", "Health"])

    def test_topic_input_validation(self):
        cases = [
            (["row_1", "row_1"], ["One", "Two"]),
            (["row_1", None], ["One", "Two"]),
            (["row_1", "speech_2"], ["One", "Two"]),
            (["row_1", "row_2"], ["One", None]),
            (["row_1", "row_2"], ["One", "  "]),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.parquet"
            for ids, text in cases:
                with self.subTest(ids=ids, text=text):
                    pd.DataFrame({"custom_id": ids, "text": text}).to_parquet(path)
                    scope = {"source_path": path}
                    exec(notebook.cells[1].source, scope)
                    with self.assertRaises(ValueError):
                        exec(notebook.cells[2].source, scope)


if __name__ == "__main__":
    unittest.main()
