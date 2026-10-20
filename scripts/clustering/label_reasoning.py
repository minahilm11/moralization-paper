"""Assign topic and style labels to reasoning traces.

Checkpoints retain responses with their source traces, prompt and settings.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = "You are a strict multi-label annotator."


def build_label_prompt(text: str, taxonomy: dict, config: dict) -> str:
    topics = ", ".join(item["id"] for item in taxonomy["topics"])
    styles = ", ".join(item["id"] for item in taxonomy["styles"])
    return f"""
Assign multi-label tags to this reasoning trace.

Allowed TOPIC ids:
[{topics}]

Allowed STYLE ids:
[{styles}]

Return strict JSON with this schema:
{{
  "topics": ["..."],
  "styles": ["..."]
}}

Rules:
- Use only allowed ids.
- Choose up to {config.get('label_top_k', 4)} topics and up to {config.get('label_top_k', 4)} styles.
- Return ids only.

Trace:
{text[:int(config.get('trace_max_chars_for_labeling', 2200))]}
""".strip()


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def exclusive_run(output_dir: Path):
    lock_path = output_dir / "run_in_progress.lock"
    try:
        handle = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError("This run is already locked. If an earlier process crashed, verify it has stopped before removing " + str(lock_path)) from exc
    try:
        with handle:
            handle.write(str(os.getpid()) + "\n")
        yield
    finally:
        lock_path.unlink()


def validate_labels(value: dict, taxonomy: dict, limit: int) -> dict:
    if not isinstance(value, dict) or set(value) != {"topics", "styles"}:
        raise ValueError("Response must contain exactly the topics and styles arrays.")
    for name in ("topics", "styles"):
        labels = value[name]
        allowed = {item["id"] for item in taxonomy[name]}
        if (not isinstance(labels, list) or len(labels) > limit
                or any(not isinstance(x, str) or x not in allowed for x in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError("Invalid, duplicated, or excessive " + name + " labels.")
    return {"topic_labels": value["topics"], "style_labels": value["styles"]}


def sanitize_label_id(value):
    label = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip().lower())
    label = re.sub(r"_+", "_", label).strip("_")
    return label[:64] if label else "label"


def extract_json_object(raw_text):
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def normalize_response(raw_text, taxonomy, limit):
    """Normalize label IDs and validate them against the taxonomy."""
    parsed = extract_json_object(raw_text or "{}")
    normalized = {}
    for kind in ("topics", "styles"):
        items, allowed = parsed.get(kind, []), {item["id"] for item in taxonomy[kind]}
        labels, seen = [], set()
        for item in items if isinstance(items, list) else []:
            if isinstance(item, str):
                label = sanitize_label_id(item)
            elif isinstance(item, dict):
                label = sanitize_label_id(item.get("id", ""))
            else:
                continue
            if label in allowed and label not in seen:
                labels.append(label)
                seen.add(label)
            if len(labels) >= limit:
                break
        normalized[kind] = labels
    return validate_labels(normalized, taxonomy, limit)


async def label_sample(sample: pd.DataFrame, taxonomy: dict, config: dict,
                       output_dir: Path, execute: bool = False) -> pd.DataFrame:
    """Label a prepared sample only with execute=True; resume only the same run.

    Use ``await label_sample(...)`` in a notebook, or asyncio.run from a CLI.
    Output directories must be empty or belong to this exact run manifest.
    """
    if execute is not True:
        raise ValueError("API labeling is disabled. Set execute=True explicitly and choose a new run output directory.")
    required = ["original_row_idx", "classification_sample_order", "rating", "reasoning_clean"]
    if sample.empty or any(c not in sample for c in required):
        raise ValueError("A nonempty prepared sample with " + ", ".join(required) + " is required.")
    data = sample.copy()
    for name in required[:3]:
        values = pd.to_numeric(data[name], errors="raise")
        if values.isna().any() or not values.mod(1).eq(0).all():
            raise ValueError(name + " must contain finite integers.")
        data[name] = values.astype("int64")
    if (not data.original_row_idx.is_unique or not data.classification_sample_order.is_unique
            or data.original_row_idx.lt(0).any() or data.classification_sample_order.lt(0).any()
            or not data.rating.between(1, 5).all()
            or not data.reasoning_clean.map(lambda x: isinstance(x, str) and bool(x.strip())).all()):
        raise ValueError("Sample IDs/order must be unique and nonnegative; ratings 1–5; traces nonempty strings.")
    data = data.sort_values("classification_sample_order").reset_index(drop=True)
    for name in ("topics", "styles"):
        items = taxonomy.get(name)
        if not isinstance(items, list) or not items:
            raise ValueError("Taxonomy requires nonempty topics and styles lists.")
        ids = [item.get("id") if isinstance(item, dict) else None for item in items]
        if (any(not isinstance(x, str) or not re.fullmatch(r"[a-z0-9_]{1,64}", x) for x in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("Taxonomy IDs must be valid and unique within " + name + ".")
    defaults = {"llm_model": "gpt-5-mini", "reasoning_effort": "medium", "label_top_k": 4,
                "trace_max_chars_for_labeling": 2200, "llm_max_concurrency": 200,
                "llm_retries": 5, "llm_retry_backoff_sec": 1.5}
    settings = {key: config.get(key, value) for key, value in defaults.items()}
    for key in ("label_top_k", "trace_max_chars_for_labeling", "llm_max_concurrency", "llm_retries"):
        if not isinstance(settings[key], int) or isinstance(settings[key], bool) or settings[key] < 1:
            raise ValueError(key + " must be a positive integer.")
    if settings["label_top_k"] > 4 or settings["llm_retry_backoff_sec"] < 0:
        raise ValueError("At most four labels and a nonnegative retry backoff are supported.")
    records = [
        {
            "original_row_idx": int(row.original_row_idx),
            "classification_sample_order": int(row.classification_sample_order),
            "rating": int(row.rating),
            "reasoning_clean": row.reasoning_clean,
        }
        for row in data.itertuples()
    ]
    prompts = {
        row["original_row_idx"]: build_label_prompt(row["reasoning_clean"], taxonomy, settings)
        for row in records
    }
    request_hashes = {}
    for key, prompt in prompts.items():
        request = {"system": SYSTEM_PROMPT, "prompt": prompt, "config": settings}
        text = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        request_hashes[key] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    identity = {
        "protocol_version": 2,
        "sample_count": len(records),
        "config": settings,
        "system_prompt": SYSTEM_PROMPT,
    }
    for field, value in (
        ("sample_sha256", records),
        ("taxonomy_sha256", taxonomy),
        ("ordered_request_sha256", list(request_hashes.items())),
    ):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        identity[field] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    text = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
    output_dir = Path(output_dir).resolve()
    manifest_path, checkpoint = output_dir / "run_manifest.json", output_dir / "checkpoint.jsonl"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not manifest_path.exists() or json.loads(manifest_path.read_text()).get("run_fingerprint") != fingerprint:
            raise ValueError("Output directory belongs to another run or has no matching manifest; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output_dir):
        completed = {}
        by_id = {r["original_row_idx"]: r for r in records}
        if checkpoint.exists():
            for line in checkpoint.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record["original_row_idx"]
                if (key not in by_id or key in completed or record["request_sha256"] != request_hashes[key]
                        or record["classification_sample_order"] != by_id[key]["classification_sample_order"]):
                    raise ValueError("Checkpoint contains an unknown, duplicate, or mismatched sample record.")
                validate_labels(
                    {"topics": record["topic_labels"], "styles": record["style_labels"]},
                    taxonomy, settings["label_top_k"],
                )
                completed[key] = record
        pending = [r for r in records if r["original_row_idx"] not in completed]
        client = None
        if pending:
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError("Set OPENAI_API_KEY only when intentionally running new API labeling.")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        manifest = dict(identity, run_fingerprint=fingerprint, status="running", completed=len(completed))
        write_manifest(manifest_path, manifest)
        queue = asyncio.Queue()
        for record in pending:
            queue.put_nowait(record)
        failures = []
        lock = asyncio.Lock()

        async def append(path, record):
            async with lock:
                with path.open("a", encoding="utf-8") as handle:
                    text = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    handle.write(text + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())

        async def worker():
            while not failures:
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                key = row["original_row_idx"]
                for attempt in range(settings["llm_retries"]):
                    raw_text, raw_response = "", None
                    try:
                        response = await client.responses.create(
                            model=settings["llm_model"], reasoning={"effort": settings["reasoning_effort"]},
                            input=[{"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                                   {"role": "user", "content": [{"type": "input_text", "text": prompts[key]}]}])
                        raw_response = response.model_dump(mode="json")
                        if getattr(response, "status", None) != "completed":
                            raise ValueError("API response did not complete successfully.")
                        raw_text = getattr(response, "output_text", "") or "\n".join(
                            getattr(c, "text", "") for item in (getattr(response, "output", []) or [])
                            for c in (getattr(item, "content", []) or []) if isinstance(getattr(c, "text", None), str))
                        labels = normalize_response(raw_text, taxonomy, settings["label_top_k"])
                        result = dict(original_row_idx=key, classification_sample_order=row["classification_sample_order"],
                                      request_sha256=request_hashes[key], raw_text=raw_text, raw_response=raw_response, **labels)
                        await append(checkpoint, result)
                        completed[key] = result
                        break
                    except Exception as exc:
                        await append(output_dir / "failed_attempts.jsonl", {
                            "original_row_idx": key, "request_sha256": request_hashes[key], "attempt": attempt + 1,
                            "error": type(exc).__name__ + ": " + str(exc), "raw_text": raw_text, "raw_response": raw_response})
                        if attempt + 1 == settings["llm_retries"]:
                            failures.append(key)
                        else:
                            await asyncio.sleep(settings["llm_retry_backoff_sec"] * 2 ** attempt)

        tasks = [asyncio.create_task(worker()) for _ in range(min(len(pending), settings["llm_max_concurrency"]))]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if client is not None:
                await client.close()
            manifest.update(status="incomplete", completed=len(completed))
            write_manifest(manifest_path, manifest)
        if failures or len(completed) != len(records):
            raise RuntimeError("Labeling is incomplete; saved " + str(len(completed)) + "/" + str(len(records)) + " records. Resume this same run after resolving the failure.")
        for name in ("topic_labels", "style_labels"):
            data[name] = [completed[int(key)][name] for key in data.original_row_idx]
        target = output_dir / "reasoning_labels.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        data.to_parquet(temporary, index=False)
        temporary.replace(target)
        manifest.update(status="complete", completed=len(data), output_file=target.name,
                        output_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        write_manifest(manifest_path, manifest)
        return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Prepared sample Parquet.")
    parser.add_argument("--output", required=True, type=Path, help="New run directory; matching interrupted runs may resume.")
    parser.add_argument("--taxonomy", required=True, type=Path, help="Existing taxonomy JSON with topics and styles.")
    parser.add_argument("--config", type=Path, help="Labeling settings JSON.")
    parser.add_argument("--allow-api", action="store_true", help="Explicitly allow paid labeling requests.")
    args = parser.parse_args()
    if not args.allow_api:
        parser.error("Paid labeling is disabled; use --allow-api to submit requests.")
    sample = pd.read_parquet(args.input)
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    result = asyncio.run(label_sample(sample, taxonomy, config, args.output, execute=True))
    print("Saved {} labeled traces to {}".format(len(result), args.output / "reasoning_labels.parquet"))


if __name__ == "__main__":
    main()
