import hashlib, json
from pathlib import Path

RAW = Path("/Users/chenyinghao/workspace/PTDBench/raw_tasks/hsz_search-r1-delivery")
T = Path("/Users/chenyinghao/workspace/PTDBench/tasks/search_r1_zero_query_vector")
report = json.loads(Path("/tmp/ptdbench_stage/evidence_report.json").read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

edited = []
for name, why in [
    ("run.py", "SHARED now resolves inside the workspace (shared/ was a sibling of the "
               "workspace in the delivery, and the harness mounts only the workspace); "
               "default interpreter is the one running the script instead of "
               "shared/runtime/.venv/bin/python, which the image now provides; --gpu "
               "lost choices=[4, 5], which were the donor host's two free cards; new "
               "--runs-dir so a graded rerun can write under $PTDBENCH_RESULTS; and "
               "trainer.default_local_dir now points at workspace/checkpoints/<run> "
               "instead of <run>/checkpoints, so pointing --runs-dir at the graded "
               "results directory does not put a 6GB actor checkpoint in it"),
    ("serve_retriever.py", "SHARED now resolves inside the workspace, same reason as run.py"),
    ("config.json", "gpu 4 -> 0 and port 8004 -> 8000: in the delivery the two "
                    "workspaces were pinned to GPU 4/port 8004 and GPU 5/port 8005, so "
                    "the numbers themselves said which half of the pair this is"),
]:
    edited.append({
        "path": f"workspace/{name}",
        "action": "packaging-adjusted",
        "why": why,
        "sha256_before": sha(RAW / "workspace_broken" / name),
        "sha256_after": sha(T / "workspace" / name),
    })

source_tree = sorted((T / "workspace/Search-R1").rglob("*.py"))
payload = {
    "schema_version": 1,
    "source": "raw_tasks/hsz_search-r1-delivery (paired delivery, 2026-10-14 runs)",
    "staged_side": "the defective workspace (delivery's workspace_broken)",
    "upstream": {
        "repo": "https://github.com/PeterGriffinJin/Search-R1.git",
        "commit": "598e61bd1d36895726d28a8d06b3a15bed19f5d3",
        "note": "recorded in the run's wandb-metadata.json; verl is vendored inside it",
    },
    "unchanged": {
        "workspace/Search-R1/": {
            "note": (
                "copied byte for byte, defect included. The only file that differs "
                "between the delivery's two workspaces is "
                "search_r1/search/retrieval_server.py (lines 117-120); every other "
                "file, the training algorithm and the scoring code are identical on "
                "both sides."
            ),
            "python_files": len(source_tree),
            "retrieval_server_sha256": sha(
                T / "workspace/Search-R1/search_r1/search/retrieval_server.py"
            ),
        },
        "workspace/shared/data/": {
            "note": "train.parquet, test.parquet and corpus.jsonl copied byte for byte; "
                    "hashes and provenance in data_manifest.json",
        },
    },
    "edited_for_packaging": edited,
    "dropped_from_the_delivery": [
        {
            "path": "shared/runtime/{pyproject.toml,uv.lock,.python-version}",
            "why": "became the environment image: docker/search-r1-runtime/ plus "
                   "docker/images/search-r1-verl.Dockerfile. The lock is still the "
                   "source of truth for the pinned stack.",
        },
        {
            "path": "shared/data/raw/*.json",
            "why": "7MB of raw HotpotQA shards the parquet was built from; their "
                   "per-shard sha256 is kept in data_manifest.json instead",
        },
        {
            "path": "shared/data/{manifest.json,retrieval_precheck.json}",
            "why": "manifest.json folded into environment/data_manifest.json; "
                   "retrieval_precheck.json compares the two sides of the pair and is "
                   "therefore oracle material, not workspace material",
        },
        {
            "path": "shared/baseline_evidence/*",
            "why": "the recorded comparison between the two sides: moved to "
                   "evaluations/solution/oracle/",
        },
        {
            "path": "workspace_fixed/*",
            "why": "the healthy twin. Its retrieval_server.py is sealed under "
                   "evaluations/solution/oracle/{code_target,format_gold}/; its run "
                   "logs and curve are in oracle/dirty_reference/",
        },
        {
            "path": "{setup_env.sh,download_models.sh,MODEL_SHA256SUMS,SHA256SUMS,README.md}",
            "why": "delivery-level tooling, replaced by environment/prepare.sh and the "
                   "image; the pinned revisions and hashes they carried are now in "
                   "prepare.sh and data_manifest.json",
        },
    ],
    "frozen_evidence": {
        "path": "workspace/results/baseline32/",
        "note": (
            "the defective run's own record, slimmed from 37MB to ~3.2MB and "
            "path-sanitized. The run lived in a directory literally named `broken` "
            "next to one named `fixed`, and those strings appeared in the logs, the "
            "experiment manifest and the W&B metadata; a diagnosis task cannot ship "
            "them. No metric value was touched."
        ),
        "substitutions": report["substitutions"],
        "substitution_counts": report["substitution_counts"],
        "files": report["files"],
    },
}
out = T / "environment/source_manifest.json"
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {out}")
