#!/usr/bin/env python3
"""Cluster reasoning embeddings with PCA, UMAP and HDBSCAN.

Sampling order: valid ratings, rating-stratified draw, trace cleaning, then
the 120-character filter. Use --sample for saved traces or --input to sample
from the full dataset.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
from pathlib import Path

CONFIG = {
    "n_samples": 50000, "sampling_strategy": "stratified_rating", "random_state": 42,
    "openai_embedding_model": "text-embedding-3-small", "embedding_dimensions": 1536,
    "openai_embedding_concurrency": 2, "openai_embedding_retries": 10,
    "openai_embedding_backoff_sec": 1.0, "openai_embedding_batch_size": 128,
    "pca_components": 100, "umap_components": 15, "umap_n_neighbors": 18,
    "umap_min_dist": 0.0, "hdbscan_grid": [[8, 3], [10, 4], [12, 4], [15, 5],
                                          [20, 7], [30, 10], [40, 12], [60, 20]],
    "max_silhouette_sample": 3000, "max_noise_for_coverage": 0.70,
    "min_clusters_for_coverage": 12, "top_terms": 14, "examples_per_cluster": 6,
    "max_clusters_for_packets": 30, "tfidf_min_df": 8, "tfidf_max_df": 0.75,
}
HISTORICAL_SAMPLE_SHA256 = "07f6fc1ee22c8468db4646f79f91d36582d8c2e1d90a745efe77442903117791"
LEADING_PATTERNS = [
    r"^to analyze the speech[^.]*\.\s*", r"^in analyzing the speech[^.]*\.\s*",
    r"^the analysis of this speech[^.]*\.\s*",
    r"^to analyze the speech from a moral foundations theory perspective[^.]*\.\s*",
    r"^to analyze the speech using moral foundations theory[^.]*\.\s*",
]
GENERIC_PHRASES = ["from a moral foundations theory perspective", "moral foundations theory",
                   "the speech", "moralizing language"]
RATING_LEAKAGE_PATTERNS = [
    r"(?i)\bi would rate\b[^.!?\n]{0,140}(?:[.!?]|$)",
    r"(?i)\bi rate\b[^.!?\n]{0,120}(?:[.!?]|$)",
    r"(?i)\bmy rating\b[^.!?\n]{0,120}(?:[.!?]|$)",
    r"(?i)\bfinal rating\b[^.!?\n]{0,120}(?:[.!?]|$)",
    r"(?i)\brating would be\b[^.!?\n]{0,120}(?:[.!?]|$)",
    r"(?i)\bmoral rating\b[^.!?\n]{0,120}(?:[.!?]|$)",
]


def clean_reasoning_trace(text):
    t = str(text).replace("\r", "\n")
    t = re.sub(r"\n\s*#+\s*", "\n", t)
    t = re.sub(r"\n\s*[-*]\s+", "\n", t)
    t = re.sub(r"\n\s*\d+[\.)]\s+", "\n", t)
    for pattern in LEADING_PATTERNS:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)
    for phrase in GENERIC_PHRASES:
        t = re.sub(re.escape(phrase), "", t, flags=re.IGNORECASE)
    for pattern in RATING_LEAKAGE_PATTERNS:
        t = re.sub(pattern, " ", t)
    return re.sub(r"\s+", " ", t).strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_fingerprint(sample):
    digest = hashlib.sha256()
    for original_id, rating, text in zip(sample.original_row_idx, sample.rating, sample.reasoning_clean):
        row = json.dumps([int(original_id), int(rating), text], ensure_ascii=False, separators=(",", ":"))
        digest.update((row + "\n").encode("utf-8"))
    return digest.hexdigest()


def original_ids(values):
    if not values.map(lambda x: isinstance(x, str) and re.fullmatch(r"row_\d+", x) is not None).all():
        raise ValueError("Source identifiers must have the form row_<original integer ID>.")
    ids = values.str.slice(4).astype("int64")
    if not ids.is_unique:
        raise ValueError("Original speech IDs must be unique.")
    return ids


def load_archived_sample(path):
    import numpy as np
    import pandas as pd

    path = Path(path)
    before = path.stat()
    sample = pd.read_parquet(path).drop(columns=["cluster"], errors="ignore")
    required = {"original_row_idx", "sample_order", "rating", "reasoning_clean"}
    if sample.empty or not required.issubset(sample.columns):
        raise ValueError("Archived sample needs original_row_idx, sample_order, rating, and reasoning_clean.")
    for column in ("original_row_idx", "sample_order", "rating"):
        numeric = pd.to_numeric(sample[column], errors="raise")
        if numeric.isna().any() or not np.isfinite(numeric).all() or not numeric.mod(1).eq(0).all():
            raise ValueError(column + " must contain finite integers.")
        sample[column] = numeric.astype("int64")
    if not sample.original_row_idx.is_unique or sample.original_row_idx.lt(0).any():
        raise ValueError("Archived original speech IDs must be unique and nonnegative.")
    if not np.array_equal(sample.sample_order.to_numpy(), np.arange(len(sample))):
        raise ValueError("Archived sample_order must be exactly 0..N-1 in file order.")
    if not sample.rating.between(1, 5).all():
        raise ValueError("Archived ratings must be integers from 1 to 5.")
    if not sample.reasoning_clean.map(lambda value: isinstance(value, str) and len(value) >= 120).all():
        raise ValueError("Archived cleaned traces must contain at least 120 characters.")
    if "new_custom_id" in sample and not np.array_equal(original_ids(sample.new_custom_id), sample.original_row_idx):
        raise ValueError("Archived new_custom_id and original_row_idx disagree.")
    fingerprint = sample_fingerprint(sample)
    file_digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Archived sample changed while loading.")
    return sample, {"input_mode": "archived_sample", "sample_file": path.name,
                    "sample_file_sha256": file_digest, "sample_rows": len(sample),
                    "rating_counts": {str(k): int(v) for k, v in sample.rating.value_counts().sort_index().items()},
                    "sample_sha256": fingerprint, "historical_sample_match": fingerprint == HISTORICAL_SAMPLE_SHA256}


def prepare_sample(source):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    source = Path(source)
    before = source.stat()
    parquet = pq.ParquetFile(source)
    columns = set(parquet.schema_arrow.names)
    id_col = next((c for c in ("new_custom_id", "custom_id") if c in columns), None)
    if {"moral_rating", "moral_reasoning"}.issubset(columns):
        rating_col, text_col = "moral_rating", "moral_reasoning"
    elif {"new_rating", "new_reasoning"}.issubset(columns):
        rating_col, text_col = "new_rating", "new_reasoning"
    else:
        raise ValueError("Input needs the final-run moral_rating/moral_reasoning or new_rating/new_reasoning pair.")
    if id_col is None:
        raise ValueError("Input needs new_custom_id or custom_id; positional row numbers are not original IDs.")
    meta_cols = list(dict.fromkeys([id_col, rating_col] + (["new_rating"] if "new_rating" in columns else [])))
    meta = pd.read_parquet(source, columns=meta_cols).reset_index(drop=True)
    meta["source_row_idx"] = np.arange(len(meta), dtype=np.int64)
    meta["original_row_idx"] = original_ids(meta[id_col])
    numeric = pd.to_numeric(meta[rating_col], errors="coerce")
    if rating_col != "new_rating" and "new_rating" in meta:
        new_rating = pd.to_numeric(meta.new_rating, errors="coerce")
        if not (numeric.eq(new_rating) | (numeric.isna() & new_rating.isna())).all():
            raise ValueError("Operational ratings differ from new_rating.")
    valid = meta[numeric.between(1, 5)].copy()
    valid["rating"] = numeric.loc[valid.index].astype(int)
    # Sample before removing empty traces to retain the sampling distribution.
    selected = valid
    if len(valid) > CONFIG["n_samples"]:
        rng = np.random.default_rng(CONFIG["random_state"])
        levels = sorted(valid.rating.unique())
        per_group = max(1, CONFIG["n_samples"] // len(levels))
        parts = []
        for level in levels:
            group = valid[valid.rating == level]
            indices = rng.choice(group.index.to_numpy(), min(per_group, len(group)), replace=False)
            parts.append(group.loc[indices])
        selected = pd.concat(parts)
    selected = selected.reset_index(drop=True)
    positions = dict(zip(selected.original_row_idx, range(len(selected))))
    rows = [None] * len(selected)
    extra = [c for c in ("speaker_name", "party_name", "date") if c in columns]
    read_cols = list(dict.fromkeys([id_col, text_col] + extra + (["new_reasoning"] if "new_reasoning" in columns else [])))
    for batch in parquet.iter_batches(batch_size=4096, columns=read_cols):
        data = batch.to_pydict()
        for index, (custom_id, text) in enumerate(zip(data[id_col], data[text_col])):
            if text_col != "new_reasoning" and "new_reasoning" in data and text != data["new_reasoning"][index]:
                raise ValueError("Operational traces differ from new_reasoning.")
            position = positions.get(int(custom_id[4:]))
            if position is None:
                continue
            record = selected.iloc[position]
            raw = str(text).strip()
            rows[position] = {"original_row_idx": int(record.original_row_idx),
                              "source_row_idx": int(record.source_row_idx), "new_custom_id": custom_id,
                              "rating": int(record.rating), "reasoning_raw": raw,
                              "reasoning_clean": clean_reasoning_trace(raw),
                              **{c: data[c][index] for c in extra}}
    if any(row is None for row in rows):
        raise ValueError("Selected source IDs were missing on the second read.")
    sample = pd.DataFrame(rows)
    if sample.empty:
        raise ValueError("No valid-rating rows were available for sampling.")
    sample = sample[sample.reasoning_clean.str.len() >= 120].reset_index(drop=True)
    sample.insert(2, "sample_order", np.arange(len(sample), dtype=np.int64))
    if sample.empty:
        raise ValueError("No traces survive the historical 120-character filter.")
    source_digest = sha256(source)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while preparing the sample.")
    fingerprint = sample_fingerprint(sample)
    return sample, {"source_file": source.name, "source_sha256": source_digest,
                    "source_rows": len(meta), "valid_rating_rows": len(valid),
                    "sampled_before_cleaning": len(selected), "dropped_after_cleaning": len(selected) - len(sample),
                    "sample_rows": len(sample), "rating_counts": {str(k): int(v) for k, v in sample.rating.value_counts().sort_index().items()},
                    "sample_sha256": fingerprint, "historical_sample_match": fingerprint == HISTORICAL_SAMPLE_SHA256}


def load_checked_cache(cache_dir, sample):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    cache_dir = Path(cache_dir)
    sample_path, vectors_path = cache_dir / "sample_for_embeddings.parquet", cache_dir / "reasoning_embeddings.npy"
    names = pq.ParquetFile(sample_path).schema_arrow.names
    key = next((c for c in ("original_row_idx", "new_custom_id", "custom_id") if c in names), None)
    if key is None or not {"reasoning_clean", "rating"}.issubset(names):
        raise ValueError("Cache must include a paired sample with original IDs, rating, and reasoning_clean.")
    paired = pd.read_parquet(sample_path, columns=[key, "rating", "reasoning_clean"])
    ids = paired[key] if key == "original_row_idx" else original_ids(paired[key])
    if (len(paired) != len(sample) or not np.array_equal(ids.to_numpy(), sample.original_row_idx.to_numpy())
            or not np.array_equal(paired.rating.to_numpy(), sample.rating.to_numpy())
            or not np.array_equal(paired.reasoning_clean.to_numpy(), sample.reasoning_clean.to_numpy())):
        raise ValueError("Cached sample IDs/order, ratings, or cleaned traces differ from the reconstructed source sample.")
    cached_config = json.loads((cache_dir / "run_config.json").read_text(encoding="utf-8"))
    cached_config = cached_config.get("config", cached_config)
    if cached_config.get("openai_embedding_model") != CONFIG["openai_embedding_model"]:
        raise ValueError("Cache metadata is missing the expected text-embedding-3-small model.")
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (len(sample), CONFIG["embedding_dimensions"]) or vectors.dtype != np.float32:
        raise ValueError("Cache must contain the expected N x 1536 float32 embeddings.")
    for start in range(0, len(vectors), 4096):
        block = vectors[start:start + 4096]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=2e-5):
            raise ValueError("Cached embeddings must be finite, nonzero, and L2-normalized.")
    digest = sha256(vectors_path)
    receipt_path = cache_dir / "embedding_manifest.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("embeddings_sha256") != digest or receipt.get("sample_sha256") != sample_fingerprint(sample):
            raise ValueError("Cache hashes differ from its embedding manifest.")
    return vectors, {"mode": "validated_cache", "model": CONFIG["openai_embedding_model"],
                     "sample_sha256": sample_fingerprint(sample), "embeddings_sha256": digest,
                     "paired_sample_file_sha256": sha256(sample_path), "shape": list(vectors.shape),
                     "dtype": str(vectors.dtype)}


async def generate_embeddings(sample):
    import numpy as np
    from openai import AsyncOpenAI
    from sklearn.preprocessing import normalize

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("--allow-api also requires OPENAI_API_KEY.")
    result = np.empty((len(sample), CONFIG["embedding_dimensions"]), dtype=np.float32)
    sem = asyncio.Semaphore(CONFIG["openai_embedding_concurrency"])
    async with AsyncOpenAI() as client:
        async def one_batch(start, texts):
            for attempt in range(CONFIG["openai_embedding_retries"]):
                try:
                    async with sem:
                        response = await client.embeddings.create(model=CONFIG["openai_embedding_model"], input=texts)
                    items = sorted(response.data, key=lambda item: item.index)
                    if [item.index for item in items] != list(range(len(texts))):
                        raise ValueError("Embedding response indices are missing or duplicated.")
                    values = np.asarray([item.embedding for item in items], dtype=np.float32)
                    if values.shape != (len(texts), result.shape[1]) or not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
                        raise ValueError("Embedding response has invalid dimensions or values.")
                    result[start:start + len(texts)] = values
                    return
                except Exception:
                    if attempt + 1 == CONFIG["openai_embedding_retries"]:
                        raise
                    await asyncio.sleep(min(60.0, CONFIG["openai_embedding_backoff_sec"] * 2 ** attempt))
        size = CONFIG["openai_embedding_batch_size"]
        tasks = [one_batch(start, sample.reasoning_clean.iloc[start:start + size].tolist()) for start in range(0, len(sample), size)]
        for completed, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            print("Embedded batch {}/{}".format(completed, len(tasks)), flush=True)
    return normalize(result)


def cluster_embeddings(sample, embeddings, output):
    import numpy as np
    import pandas as pd
    import umap
    import hdbscan
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(sample) < CONFIG["pca_components"]:
        raise ValueError("Historical PCA requires at least 100 sampled traces.")
    pca = PCA(n_components=min(CONFIG["pca_components"], embeddings.shape[1]), random_state=CONFIG["random_state"])
    reduced = pca.fit_transform(embeddings)
    matrix = umap.UMAP(n_components=CONFIG["umap_components"], n_neighbors=CONFIG["umap_n_neighbors"],
                       min_dist=CONFIG["umap_min_dist"], metric="cosine", random_state=CONFIG["random_state"],
                       n_jobs=1).fit_transform(reduced)
    evaluations = []
    for minimum, neighbors in CONFIG["hdbscan_grid"]:
        labels = hdbscan.HDBSCAN(min_cluster_size=minimum, min_samples=neighbors, metric="euclidean").fit_predict(matrix)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = float(np.mean(labels == -1))
        idx = np.flatnonzero(labels != -1)
        if len(idx) > CONFIG["max_silhouette_sample"]:
            idx = np.random.default_rng(CONFIG["random_state"]).choice(idx, CONFIG["max_silhouette_sample"], replace=False)
        silhouette = float(silhouette_score(matrix[idx], labels[idx])) if 1 < len(set(labels[idx])) < len(idx) else float("nan")
        evaluations.append({"min_cluster_size": minimum, "min_samples": neighbors, "n_clusters": n_clusters,
                            "noise_ratio": noise, "silhouette_non_noise": silhouette,
                            "coverage_score": n_clusters * max(0.0, 1.0 - noise)})
    evaluation = pd.DataFrame(evaluations).sort_values("coverage_score", ascending=False).reset_index(drop=True)
    eligible = evaluation[(evaluation.noise_ratio <= CONFIG["max_noise_for_coverage"]) & (evaluation.n_clusters >= CONFIG["min_clusters_for_coverage"])]
    best = eligible.iloc[0] if len(eligible) else evaluation.iloc[0]
    result = sample.copy()
    result["cluster"] = hdbscan.HDBSCAN(min_cluster_size=int(best.min_cluster_size), min_samples=int(best.min_samples), metric="euclidean").fit_predict(matrix)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), min_df=CONFIG["tfidf_min_df"],
        max_df=CONFIG["tfidf_max_df"], stop_words="english"
    )
    tfidf = vectorizer.fit_transform(result.reasoning_clean)
    terms, summaries = np.array(vectorizer.get_feature_names_out()), []
    for cluster, count in result[result.cluster != -1].cluster.value_counts().items():
        mask = result.cluster.to_numpy() == cluster
        inside = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        outside = np.asarray(tfidf[~mask].mean(axis=0)).ravel() if (~mask).any() else np.zeros_like(inside)
        top = terms[np.argsort(inside - outside)[::-1][:CONFIG["top_terms"]]]
        summaries.append({"cluster": int(cluster), "size": int(count), "avg_rating": round(float(result.loc[mask, "rating"].mean()), 3), "top_terms": ", ".join(top)})
    summary = pd.DataFrame(summaries, columns=["cluster", "size", "avg_rating", "top_terms"]).sort_values("size", ascending=False).reset_index(drop=True)
    rng, packets = np.random.default_rng(CONFIG["random_state"]), []
    for _, row in summary.head(CONFIG["max_clusters_for_packets"]).iterrows():
        indices = result.index[result.cluster == int(row.cluster)].to_numpy()
        selected = rng.choice(indices, min(CONFIG["examples_per_cluster"], len(indices)), replace=False)
        packets.append({"cluster": int(row.cluster), "size": int(row["size"]), "avg_rating": float(row.avg_rating),
                        "top_terms": row.top_terms, "sample_original_row_idx": result.loc[selected, "original_row_idx"].tolist(),
                        "sample_traces": [result.loc[i, "reasoning_clean"][:450] for i in selected]})
    evaluation.to_csv(output / "cluster_eval.csv", index=False)
    summary.to_csv(output / "cluster_summary.csv", index=False)
    result.to_parquet(output / "cluster_assignments_with_original_idx.parquet", index=False)
    result.drop(columns=["reasoning_raw"], errors="ignore").to_parquet(output / "cluster_assignments.parquet", index=False)
    (output / "cluster_packets.json").write_text(json.dumps({"packets": packets}, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(output / "pca_coordinates.npy", reduced)
    np.save(output / "umap_coordinates.npy", matrix)
    return {"selected_hdbscan": {k: (None if pd.isna(v) else float(v)) for k, v in best.items()},
            "pca_solver_used": getattr(pca, "_fit_svd_solver", "unknown"),
            "nonnoise_clusters": int(result.loc[result.cluster != -1, "cluster"].nunique()),
            "noise_rows": int(result.cluster.eq(-1).sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Full final-run input Parquet; reconstruct the sample.")
    inputs.add_argument("--sample", type=Path, help="Cleaned sample, e.g. data/reasoning/cluster_assignments_with_original_idx.parquet.")
    parser.add_argument("--output", required=True, type=Path, help="New or empty directory; existing files are never replaced.")
    parser.add_argument("--cache-dir", type=Path, help="Read-only directory with paired sample, embeddings, and run_config.json.")
    parser.add_argument("--prepare-only", action="store_true", help="Reconstruct sample and optionally validate cache; no API or clustering.")
    parser.add_argument("--expect-historical", action="store_true", help="Require the exact 42,843 historical IDs/order/ratings/text fingerprint.")
    parser.add_argument("--allow-api", action="store_true", help="Explicitly permit paid embedding requests when --cache-dir is absent.")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be new or empty; choose a different directory.")
    if args.cache_dir and output == args.cache_dir.resolve():
        parser.error("--output must differ from --cache-dir.")
    if not (args.input or args.sample).is_file():
        parser.error("Selected input/sample file does not exist.")
    if not args.prepare_only and not args.cache_dir and not args.allow_api:
        parser.error("Provide --cache-dir, or explicitly --allow-api; --prepare-only works entirely offline.")
    if not args.prepare_only:
        # Fail on missing scientific dependencies before any paid requests.
        import sklearn, umap, hdbscan  # noqa: F401
    sample, stats = prepare_sample(args.input) if args.input else load_archived_sample(args.sample)
    print(json.dumps(stats, indent=2), flush=True)
    if args.expect_historical and not stats["historical_sample_match"]:
        raise ValueError("Prepared sample differs from the historical paper sample; no outputs written.")
    if not args.prepare_only and len(sample) < CONFIG["pca_components"]:
        raise ValueError("Historical PCA requires at least 100 sampled traces; no API requests made.")
    embeddings, receipt = (None, None)
    if args.cache_dir:
        embeddings, receipt = load_checked_cache(args.cache_dir, sample)
        print("Cache passed exact paired-sample, model, vector, and hash checks.", flush=True)
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "umap-learn", "hdbscan", "openai"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {"config": CONFIG, "sample": stats, "versions": versions, "script_sha256": sha256(Path(__file__)),
                "cache_validation": receipt, "status": "prepared", "api_allowed": args.allow_api}
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two runs from claiming the same directory.
    with (output / "run_config.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    sample.to_parquet(output / "sample_for_embeddings.parquet", index=False)
    if args.prepare_only:
        print("Offline preparation complete: " + str(output), flush=True)
        return 0
    import numpy as np
    if args.cache_dir:
        shutil.copyfile(args.cache_dir / "reasoning_embeddings.npy", output / "reasoning_embeddings.npy")
        if sha256(output / "reasoning_embeddings.npy") != receipt["embeddings_sha256"]:
            raise ValueError("Embedding cache changed while copying; the snapshot is not validated.")
        embeddings = np.load(output / "reasoning_embeddings.npy", mmap_mode="r", allow_pickle=False)
    else:
        if not args.allow_api:
            raise RuntimeError("API permission is required.")
        embeddings = asyncio.run(generate_embeddings(sample))
        np.save(output / "reasoning_embeddings.npy", embeddings)
        receipt = {"mode": "new_api_embeddings", "model": CONFIG["openai_embedding_model"],
                   "sample_sha256": stats["sample_sha256"], "shape": list(embeddings.shape), "dtype": str(embeddings.dtype)}
    receipt = dict(receipt, embeddings_sha256=sha256(output / "reasoning_embeddings.npy"))
    (output / "embedding_manifest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    manifest.update(cluster_embeddings(sample, embeddings, output))
    manifest["status"] = "complete"
    (output / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Clustering complete: " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
