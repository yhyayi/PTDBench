import hashlib, json
from pathlib import Path

RAW = Path("/Users/chenyinghao/workspace/PTDBench/raw_tasks/hsz_search-r1-delivery")
T = Path("/Users/chenyinghao/workspace/PTDBench/tasks/search_r1_zero_query_vector")
src = json.loads((RAW / "shared/data/manifest.json").read_text())

def sha(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

shipped = {}
for name in ("train.parquet", "test.parquet", "corpus.jsonl"):
    p = T / "workspace/shared/data" / name
    shipped[name] = {"sha256": sha(p), "size": p.stat().st_size}
    assert shipped[name]["sha256"] == src["sha256"][name], name

corpus_rows = sum(1 for _ in (T / "workspace/shared/data/corpus.jsonl").open())
payload = {
    "schema_version": 1,
    "dataset": src["dataset"],
    "config": src["config"],
    "selection": src["selection"],
    "counts": src["counts"],
    "corpus_passages": corpus_rows,
    "shipped_in_workspace": {
        "shared/data/train.parquet": shipped["train.parquet"],
        "shared/data/test.parquet": shipped["test.parquet"],
        "shared/data/corpus.jsonl": shipped["corpus.jsonl"],
    },
    "built_at_prepare_time": {
        "shared/data/e5_Flat.index": {
            "builder": "environment/build_index.py",
            "encoder": "intfloat/e5-base-v2 @ f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
            "faiss_type": "Flat",
            "metric": "METRIC_INNER_PRODUCT",
            "pooling": "mean, then L2-normalised",
            "document_prefix": "passage: ",
            "max_length": 256,
            "batch_size": 512,
            "expected_shape": [corpus_rows, 768],
            "recorded_sha256": "790cb5bc76b4fe8712ad785ce8f2cfb80a5bdf44d93d58a69dc7684f7e747f7a",
            "recorded_sha256_note": (
                "sha256 of the index the recorded runs used (results/baseline32/"
                "experiment.json input_sha256). A rebuild is semantically the same "
                "index but not bitwise identical: batch composition, fp16-vs-fp32 "
                "accumulation and the GPU matmul kernel all move the last bits. Do "
                "not gate prepare on this value."
            ),
        }
    },
    "upstream_sources": src["sources"],
    "note": (
        "HotpotQA distractor, first 1024 rows of the official train split and first "
        "128 rows of the official validation split, no outcome-based filtering. The "
        "corpus is the deduplicated context paragraphs of those rows, so every "
        "question's supporting documents are retrievable; validation questions and "
        "answers are not training samples. The raw per-shard JSON the parquet was "
        "built from is not shipped -- the per-shard sha256 under upstream_sources is "
        "its provenance record."
    ),
}
out = T / "environment/data_manifest.json"
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {out}, {len(payload['upstream_sources'])} source shards, corpus rows {corpus_rows}")
