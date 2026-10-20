#!/usr/bin/env python3
"""Run the new moralization prompt with the OpenAI Batch API.

The script prepares JSONL batch inputs, submits them, downloads completed
outputs, parses <thinking>/<answer> tags, and merges completed batch chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from noshot_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def extract_tag(tag: str, text: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_rating(content: str) -> int | None:
    answer = extract_tag("answer", content)
    match = re.search(r"\b([1-5])\b", answer)
    if match:
        return int(match.group(1))
    fallback = re.search(r"<answer>.*?([1-5]).*?</answer>", content, flags=re.DOTALL | re.IGNORECASE)
    return int(fallback.group(1)) if fallback else None


def openai_client() -> Any:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before calling submit, status, or download.")
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install openai before calling submit, status, or download.") from exc
    return OpenAI()


def response_text(file_response: Any) -> str:
    if hasattr(file_response, "text"):
        return file_response.text
    if hasattr(file_response, "content"):
        content = file_response.content
        return content.decode("utf-8") if isinstance(content, bytes) else str(content)
    if hasattr(file_response, "read"):
        content = file_response.read()
        return content.decode("utf-8") if isinstance(content, bytes) else str(content)
    return str(file_response)


def prepare_batch(args: argparse.Namespace) -> None:
    out_dir = args.output_dir / f"batch_{args.start}_{args.stop}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / f"manifest_{args.start}_{args.stop}.parquet"
    input_path = out_dir / f"batch_input_{args.start}_{args.stop}.jsonl"
    config_path = out_dir / "run_config.json"

    if not args.overwrite and (manifest_path.exists() or input_path.exists()):
        raise FileExistsError(f"{out_dir} already contains prepared input. Use --overwrite to replace it.")

    df = pd.read_parquet(args.input_path)
    if args.text_column not in df.columns:
        raise ValueError(f"Input data does not contain text column: {args.text_column}")

    selected = df.iloc[args.start : args.stop].copy()
    selected.insert(0, "row_idx", selected.index.astype(int))
    selected = selected.reset_index(drop=True)
    selected.to_parquet(manifest_path, index=False)

    with input_path.open("w", encoding="utf-8") as handle:
        for _, row in selected.iterrows():
            row_idx = int(row["row_idx"])
            speech = "" if pd.isna(row[args.text_column]) else str(row[args.text_column])
            request = {
                "custom_id": f"row_{row_idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(speech=speech)},
                    ],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "seed": args.seed,
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")

    config = {
        "input_path": str(args.input_path),
        "text_column": args.text_column,
        "start": args.start,
        "stop": args.stop,
        "rows": len(selected),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "input_jsonl": str(input_path),
        "manifest": str(manifest_path),
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(selected):,} requests in {out_dir}")


def submit_batch(args: argparse.Namespace) -> None:
    if args.prepare_first:
        prepare_batch(args)
    out_dir = args.output_dir / f"batch_{args.start}_{args.stop}"
    input_path = out_dir / f"batch_input_{args.start}_{args.stop}.jsonl"
    batch_info_path = out_dir / "batch_info.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Prepare the batch first: {input_path}")
    if batch_info_path.exists() and not args.force:
        raise FileExistsError(f"{batch_info_path} already exists. Use --force to submit another batch.")

    client = openai_client()
    with input_path.open("rb") as handle:
        uploaded_file = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_info = {
        "batch_id": batch.id,
        "input_file_id": uploaded_file.id,
        "start": args.start,
        "stop": args.stop,
        "status": batch.status,
        "created_at": batch.created_at,
        "full_response": batch.model_dump(),
    }
    batch_info_path.write_text(json.dumps(batch_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Submitted {batch.id}; metadata saved to {batch_info_path}")


def show_status(args: argparse.Namespace) -> Any:
    out_dir = args.output_dir / f"batch_{args.start}_{args.stop}"
    batch_info_path = out_dir / "batch_info.json"
    if not batch_info_path.exists():
        raise FileNotFoundError(f"No batch_info.json found in {out_dir}")
    info = json.loads(batch_info_path.read_text(encoding="utf-8"))
    batch = openai_client().batches.retrieve(info["batch_id"])
    (out_dir / "batch_status_latest.json").write_text(
        json.dumps(batch.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{batch.id}: {batch.status}")
    print(f"request_counts: {batch.request_counts}")
    print(f"output_file_id: {batch.output_file_id}")
    print(f"error_file_id: {batch.error_file_id}")
    return batch


def download_batch(args: argparse.Namespace) -> None:
    out_dir = args.output_dir / f"batch_{args.start}_{args.stop}"
    batch_info_path = out_dir / "batch_info.json"
    if not batch_info_path.exists():
        raise FileNotFoundError(f"No batch_info.json found in {out_dir}")
    info = json.loads(batch_info_path.read_text(encoding="utf-8"))
    batch = show_status(args)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} is {batch.status}, not completed.")
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch.id} completed without an output_file_id.")

    client = openai_client()
    output = response_text(client.files.content(batch.output_file_id))
    results_path = out_dir / "batch_results.jsonl"
    results_path.write_text(output, encoding="utf-8")

    if batch.error_file_id:
        errors = response_text(client.files.content(batch.error_file_id))
        (out_dir / "batch_errors.jsonl").write_text(errors, encoding="utf-8")

    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record.get("custom_id", "")
        row_idx = int(custom_id.split("_", 1)[1]) if custom_id.startswith("row_") else None
        error = record.get("error")
        body = record.get("response", {}).get("body", {}) if error is None else {}
        content = body["choices"][0]["message"]["content"] if body.get("choices") else ""
        rows.append(
            {
                "row_idx": row_idx,
                "custom_id": custom_id,
                "new_rating": extract_rating(content) if content else None,
                "new_reasoning": extract_tag("thinking", content) if content else "",
                "raw_response": content,
                "error": json.dumps(error, ensure_ascii=False) if error else "",
                "usage": json.dumps(body.get("usage", {}), ensure_ascii=False),
            }
        )

    manifest = pd.read_parquet(out_dir / f"manifest_{info['start']}_{info['stop']}.parquet")
    parsed = pd.DataFrame(rows)
    repeated_output_cols = [col for col in parsed.columns if col != "row_idx" and col in manifest.columns]
    if repeated_output_cols:
        manifest = manifest.drop(columns=repeated_output_cols)
    merged = manifest.merge(parsed, on="row_idx", how="left")
    parquet_path = out_dir / f"ratings_{info['start']}_{info['stop']}.parquet"
    csv_path = out_dir / f"ratings_{info['start']}_{info['stop']}.csv"
    merged.to_parquet(parquet_path, index=False)
    merged.to_csv(csv_path, index=False)
    print(f"Saved parsed rows to {parquet_path}")


def merge_batches(args: argparse.Namespace) -> None:
    frames = []
    for path in sorted(args.output_dir.glob("batch_*/ratings_*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No parsed batch parquet files found under {args.output_dir}")

    merged = pd.concat(frames, ignore_index=True).sort_values("row_idx")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "new_prompt_noshot_4omini_ratings_all_available.parquet"
    csv_path = args.output_dir / "new_prompt_noshot_4omini_ratings_all_available.csv"
    merged.to_parquet(parquet_path, index=False)
    merged.to_csv(csv_path, index=False)

    ratings = pd.to_numeric(merged["new_rating"], errors="coerce")
    summary = {
        "rows": int(len(merged)),
        "missing_ratings": int(ratings.isna().sum()),
        "rating_counts": {str(i): int((ratings == i).sum()) for i in range(1, 6)},
        "mean_rating": float(ratings.mean()),
        "merged_at_unix": int(time.time()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Merged {len(merged):,} rows to {parquet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "submit", "status", "download", "merge"])
    parser.add_argument("--input-path", type=Path, default=Path("data/speeches_1990_2022.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/noshot_4omini_batches"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=25000)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-first", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare_batch(args)
    elif args.command == "submit":
        submit_batch(args)
    elif args.command == "status":
        show_status(args)
    elif args.command == "download":
        download_batch(args)
    elif args.command == "merge":
        merge_batches(args)
