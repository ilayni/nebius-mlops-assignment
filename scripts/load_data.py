"""
Download BIRD dev set and prepare evaluation files.

Usage:
    python scripts/load_data.py

Creates:
    data/bird/<db_id>.sqlite     — one SQLite file per database
    evals/eval_set.jsonl         — 30 questions (random seed=0)
    load_test/perf_pool.jsonl    — 1500 questions
"""
import json
import os
import random
import shutil
import sqlite3
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "bird"
EVALS_DIR = ROOT / "evals"
LOAD_TEST_DIR = ROOT / "load_test"
BIRD_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
ZIP_PATH = ROOT / "data" / "dev.zip"

EVAL_SIZE = 30
PERF_POOL_SIZE = 1500


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading BIRD dev set (~500MB)...")
        urllib.request.urlretrieve(BIRD_URL, ZIP_PATH)
        print("Download complete.")
    else:
        print("ZIP already present, skipping download.")


def extract():
    extract_dir = ROOT / "data" / "dev_extracted"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(extract_dir)
    return extract_dir


def consolidate_dbs(extract_dir: Path):
    print("Consolidating SQLite files...")
    for sqlite_file in extract_dir.rglob("*.sqlite"):
        dest = DATA_DIR / sqlite_file.name
        if not dest.exists():
            shutil.copy2(sqlite_file, dest)
    print(f"Databases in {DATA_DIR}: {len(list(DATA_DIR.glob('*.sqlite')))}")


def load_questions(extract_dir: Path) -> list[dict]:
    questions_file = next(extract_dir.rglob("dev.json"), None)
    if questions_file is None:
        raise FileNotFoundError("dev.json not found in extracted BIRD archive")
    with open(questions_file) as f:
        return json.load(f)


def write_jsonl(questions: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")


def main():
    download()
    extract_dir = extract()
    consolidate_dbs(extract_dir)

    all_questions = load_questions(extract_dir)
    normalized = []
    for q in all_questions:
        normalized.append({
            "question_id": q.get("question_id", ""),
            "db_id": q.get("db_id", ""),
            "question": q.get("question", ""),
            "gold_sql": q.get("SQL", q.get("gold_sql", "")),
        })

    rng = random.Random(0)
    sample = rng.sample(normalized, min(EVAL_SIZE, len(normalized)))
    write_jsonl(sample, EVALS_DIR / "eval_set.jsonl")
    print(f"Wrote {len(sample)} questions to evals/eval_set.jsonl")

    perf_pool = rng.sample(normalized, min(PERF_POOL_SIZE, len(normalized)))
    write_jsonl(perf_pool, LOAD_TEST_DIR / "perf_pool.jsonl")
    print(f"Wrote {len(perf_pool)} questions to load_test/perf_pool.jsonl")


if __name__ == "__main__":
    main()
