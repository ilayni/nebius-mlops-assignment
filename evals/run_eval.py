"""
BIRD-bench evaluation harness.

Usage:
    # Start agent server first:
    uvicorn agent.server:app --host 0.0.0.0 --port 8001

    python evals/run_eval.py
    python evals/run_eval.py --output results/eval_baseline.json
    python evals/run_eval.py --dry-run    # skip LLM, verify scoring logic only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

EVAL_SET = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "eval_baseline.json"
AGENT_URL = "http://localhost:8001/answer"


def run_sql(db_id: str, sql: str):
    from agent.execution import execute_sql
    result = execute_sql(db_id, sql)
    return result.ok, result.rows, result.error


def canonicalize(rows):
    if rows is None:
        return None
    return sorted(tuple(str(c) for c in row) for row in rows)


def matches(gold_rows, pred_rows) -> bool:
    return canonicalize(gold_rows) == canonicalize(pred_rows)


def eval_one(question: dict, agent_url: str, dry_run: bool = False) -> dict:
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    q_text = question["question"]

    t0 = time.monotonic()
    error = None
    agent_data = {}

    if not dry_run:
        try:
            resp = httpx.post(
                agent_url,
                json={"question": q_text, "db": db_id},
                timeout=120.0,
            )
            resp.raise_for_status()
            agent_data = resp.json()
        except Exception as e:
            error = str(e)

    latency = round(time.monotonic() - t0, 3)

    gold_ok, gold_rows, _ = run_sql(db_id, gold_sql)

    history = agent_data.get("history", [])
    iter_sqls = [
        h["sql"]
        for h in history
        if h.get("node") in ("generate_sql", "revise") and "sql" in h
    ]

    per_iter_match = []
    for sql in iter_sqls:
        iter_ok, iter_rows, _ = run_sql(db_id, sql)
        per_iter_match.append(
            matches(gold_rows, iter_rows) if (gold_ok and iter_ok) else False
        )

    predicted_sql = agent_data.get("sql", "")
    agent_ok = agent_data.get("ok", False)

    if agent_ok and predicted_sql and gold_ok and not dry_run:
        pred_ok, pred_rows, _ = run_sql(db_id, predicted_sql)
        final_match = matches(gold_rows, pred_rows) if pred_ok else False
    else:
        final_match = False

    return {
        "db_id": db_id,
        "question": q_text,
        "gold_sql": gold_sql,
        "predicted_sql": predicted_sql,
        "match": final_match,
        "iterations": agent_data.get("iterations", 0),
        "latency_s": latency,
        "error": error or agent_data.get("error"),
        "per_iter_match": per_iter_match,
        "history": [h.get("node") for h in history],
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}

    overall = sum(1 for r in results if r["match"]) / n

    max_k = max((len(r.get("per_iter_match", [])) for r in results), default=0)
    per_iter = []
    for k in range(max(max_k, 1)):
        correct = 0
        for r in results:
            pim = r.get("per_iter_match", [])
            if k < len(pim):
                if pim[k]:
                    correct += 1
            elif pim:
                if pim[-1]:
                    correct += 1
        per_iter.append(round(correct / n, 3))

    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    return {
        "n": n,
        "overall_pass_rate": round(overall, 3),
        "pass_rate_by_iteration": per_iter,
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "--output", default=str(DEFAULT_OUTPUT), dest="output")
    parser.add_argument("--agent-url", default=AGENT_URL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    questions = []
    with open(EVAL_SET) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"Running eval on {len(questions)} questions (dry_run={args.dry_run})")
    t_start = time.monotonic()
    results = []
    for i, q in enumerate(questions):
        r = eval_one(q, args.agent_url, dry_run=args.dry_run)
        status = "+" if r["match"] else "x"
        print(f"  [{i+1:02d}/{len(questions)}] {status} {r['db_id']} -- {r['iterations']} iter, {r['latency_s']}s")
        results.append(r)

    wall_clock = round(time.monotonic() - t_start, 1)
    summary = summarize(results)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "wall_clock_seconds": wall_clock, "results": results}, f, indent=2)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
