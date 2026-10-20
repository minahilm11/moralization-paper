"""Offline validation and reconstruction of the published reasoning sample.

Historical cleaning and seed-42 random sampling, with stable speech IDs.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def clean_reasoning_trace(text: str) -> str:
    """Historical classifier cleaner, including its incomplete score removal."""
    t = str(text).replace("\r", "\n")
    t = re.sub(r"\n\s*#+\s*", "\n", t)
    t = re.sub(r"\n\s*[-*]\s+", "\n", t)
    t = re.sub(r"\n\s*\d+[\.)]\s+", "\n", t)

    patterns = [
        r"(?i)\bi would rate\b[^.!?\n]{0,140}(?:[.!?]|$)",
        r"(?i)\bi rate\b[^.!?\n]{0,120}(?:[.!?]|$)",
        r"(?i)\bmy rating\b[^.!?\n]{0,120}(?:[.!?]|$)",
        r"(?i)\bfinal rating\b[^.!?\n]{0,120}(?:[.!?]|$)",
        r"(?i)\brating would be\b[^.!?\n]{0,120}(?:[.!?]|$)",
        r"(?i)\bmoral rating\b[^.!?\n]{0,120}(?:[.!?]|$)",
    ]
    for p in patterns:
        t = re.sub(p, " ", t)

    t = re.sub(r"\s+", " ", t).strip()
    return t


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_column(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.isfinite(numeric).all() or not numeric.mod(1).eq(0).all():
        raise ValueError(name + " must contain finite integers.")
    return numeric.astype("int64")


def _taxonomy_ids(taxonomy: dict, kind: str, expected_count: int) -> set:
    entries = taxonomy.get(kind)
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError("Expected " + str(expected_count) + " taxonomy " + kind + ".")
    ids = [entry.get("id") if isinstance(entry, dict) else None for entry in entries]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Taxonomy IDs must be nonempty unique strings in " + kind + ".")
    return set(ids)


def load_labeled_sample(path: Path, taxonomy: dict, expected_rows: int = 50000) -> pd.DataFrame:
    """Load and validate the saved 50k artifact; label arrays become lists.

    Topic lists may be empty (two historical rows have no topic label). Every
    published row has at least one style label. Sample position is never used
    as a substitute for original speech identity.
    """
    sample = pd.read_parquet(path)
    required = {"original_row_idx", "classification_sample_order", "rating", "reasoning_clean",
                "topic_labels", "style_labels"}
    if not required.issubset(sample.columns):
        raise ValueError("saved sample is missing columns: " + ", ".join(sorted(required - set(sample.columns))))
    if len(sample) != expected_rows:
        raise ValueError("The published saved sample must contain exactly 50,000 rows.")
    for column in ("original_row_idx", "classification_sample_order", "rating"):
        sample[column] = _integer_column(sample[column], column)
    if not sample.original_row_idx.is_unique or sample.original_row_idx.lt(0).any():
        raise ValueError("Original speech IDs must be unique nonnegative integers.")
    if not np.array_equal(sample.classification_sample_order.to_numpy(), np.arange(len(sample))):
        raise ValueError("Stored sample order must be exactly 0..49,999 in file order.")
    if not sample.rating.between(1, 5).all():
        raise ValueError("Ratings must be between 1 and 5.")
    if not sample.reasoning_clean.map(lambda value: isinstance(value, str) and len(value) >= 120).all():
        raise ValueError("Cleaned traces must be strings of at least 120 characters.")

    for column, kind, count in (("topic_labels", "topics", 62), ("style_labels", "styles", 23)):
        allowed = _taxonomy_ids(taxonomy, kind, count)

        def normalize_labels(value):
            if isinstance(value, np.ndarray):
                if value.ndim != 1:
                    raise ValueError(column + " must contain one-dimensional label arrays.")
                value = value.tolist()
            elif isinstance(value, tuple):
                value = list(value)
            if not isinstance(value, list):
                raise ValueError(column + " must contain lists or arrays.")
            if (len(value) > 4 or (kind == "styles" and not value)
                    or any(not isinstance(item, str) or item not in allowed for item in value)):
                raise ValueError("Invalid or excessive labels in " + column + ".")
            if len(set(value)) != len(value):
                raise ValueError("Duplicate labels within a row of " + column + ".")
            return list(value)

        sample[column] = sample[column].map(normalize_labels)

    return sample


def _original_id(custom_id) -> int:
    if not isinstance(custom_id, str) or not re.fullmatch(r"row_\d+", custom_id):
        raise ValueError("Invalid original speech identifier: " + repr(custom_id))
    return int(custom_id[4:])


def reconstruct_sample(source_path: Path, config: dict):
    """Reconstruct the historical random sample with two bounded-memory passes.

    The first pass retains only eligible IDs. The second keeps text for selected
    rows only, avoiding materializing the full 1.5 GB source dataframe. Source
    order is preserved before pandas' random sample, exactly as historically.
    """
    source_path = Path(source_path)
    if str(config.get("sampling_strategy", "random")).lower() != "random":
        raise ValueError("This reconstruction implements the published random sample, not stratification.")
    n_samples = config.get("n_samples", 50000)
    seed = config.get("random_state", 42)
    if n_samples is not None and (isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1):
        raise ValueError("n_samples must be a positive integer or None.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("random_state must be an integer.")
    initial_stat = source_path.stat()
    parquet = pq.ParquetFile(source_path)
    available = set(parquet.schema_arrow.names)
    columns = ["new_custom_id", "moral_reasoning", "moral_rating"]
    if not set(columns).issubset(available):
        raise ValueError("Source must contain " + ", ".join(columns) + ".")
    # Present in the actual analysis input: verify that its operational columns
    # still point to the new run, rather than accidentally selecting old ratings.
    checks = [name for name in ("new_reasoning", "new_rating") if name in available]
    source_rows = invalid_rating = missing_reasoning = too_short = 0
    eligible_ids = []
    seen = set()
    for batch in parquet.iter_batches(batch_size=4096, columns=columns + checks):
        data = batch.to_pydict()
        ratings = pd.to_numeric(pd.Series(data["moral_rating"]), errors="coerce")
        if "new_rating" in data:
            new_ratings = pd.to_numeric(pd.Series(data["new_rating"]), errors="coerce")
            if not (ratings.eq(new_ratings) | (ratings.isna() & new_ratings.isna())).all():
                raise ValueError("Operational moral_rating differs from new_rating.")
        for position, (custom_id, text, rating) in enumerate(zip(data["new_custom_id"], data["moral_reasoning"], ratings)):
            source_rows += 1
            original_id = _original_id(custom_id)
            if original_id in seen:
                raise ValueError("Duplicate original speech ID in source: " + str(original_id))
            seen.add(original_id)
            if "new_reasoning" in data and text != data["new_reasoning"][position]:
                raise ValueError("Operational moral_reasoning differs from new_reasoning.")
            if pd.isna(text):
                missing_reasoning += 1
                continue
            if pd.isna(rating) or not 1 <= rating <= 5:
                invalid_rating += 1
                continue
            if len(clean_reasoning_trace(str(text))) < 120:
                too_short += 1
                continue
            eligible_ids.append(original_id)

    selected = pd.Series(eligible_ids, dtype="int64")
    if n_samples is not None and len(selected) > n_samples:
        selected = selected.sample(n=n_samples, random_state=seed)
    selected_ids = selected.to_numpy()
    positions = {int(original_id): position for position, original_id in enumerate(selected_ids)}
    rows = [None] * len(selected_ids)
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        data = batch.to_pydict()
        for custom_id, text, rating in zip(data["new_custom_id"], data["moral_reasoning"], data["moral_rating"]):
            original_id = _original_id(custom_id)
            position = positions.get(original_id)
            if position is not None:
                rows[position] = (original_id, position, int(pd.to_numeric(rating)), clean_reasoning_trace(str(text)))
    if any(row is None for row in rows):
        raise ValueError("A selected speech was missing on the second source pass.")
    sample = pd.DataFrame(rows, columns=["original_row_idx", "classification_sample_order", "rating", "reasoning_clean"])
    source_digest = sha256(source_path)
    final_stat = source_path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError("Source file changed during sample reconstruction.")
    expected_digest = config.get("expected_source_sha256")
    if expected_digest is not None and source_digest != expected_digest:
        raise ValueError("Source SHA256 differs from expected_source_sha256.")
    stats = {
        "source_path": source_path.name, "source_sha256": source_digest,
        "source_rows": source_rows, "eligible_rows": len(eligible_ids),
        "excluded_rows": source_rows - len(eligible_ids), "sample_rows": len(sample),
        "excluded_missing_reasoning": missing_reasoning, "excluded_invalid_rating": invalid_rating,
        "excluded_clean_lt_120": too_short, "sampling_strategy": "random", "random_state": seed,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    return sample, stats


def assert_sample_matches(rebuilt: pd.DataFrame, saved: pd.DataFrame) -> dict:
    """Require exact original IDs/order, ratings, and cleaned text; return hashes."""
    columns = ["original_row_idx", "classification_sample_order", "rating", "reasoning_clean"]
    for name, frame in (("rebuilt", rebuilt), ("saved", saved)):
        if not set(columns).issubset(frame.columns):
            raise ValueError(name + " sample lacks required identity/rating/text columns.")
    if len(rebuilt) != len(saved):
        raise ValueError("Rebuilt and saved sample lengths differ.")
    for column in columns:
        left = rebuilt[column].reset_index(drop=True)
        right = saved[column].reset_index(drop=True)
        if not left.eq(right).all():
            mismatches = int((~left.eq(right)).sum())
            raise ValueError("Rebuilt sample differs in " + column + " (" + str(mismatches) + " rows).")

    return {
        "sample_rows": len(rebuilt),
        "ordered_id_sha256": hashlib.sha256(
            "\n".join("row_" + str(int(value)) for value in rebuilt.original_row_idx).encode("utf-8")
        ).hexdigest(),
        "cleaned_text_sha256": hashlib.sha256(
            "\n".join(rebuilt.reasoning_clean).encode("utf-8")
        ).hexdigest(),
        "rating_sha256": hashlib.sha256(
            "\n".join(str(int(value)) for value in rebuilt.rating).encode("utf-8")
        ).hexdigest(),
    }
