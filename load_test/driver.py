"""Async load driver for the agent endpoint.

Samples questions from load_test/perf_pool.jsonl and fires them at the
agent at the requested RPS for the requested duration, recording per-
request latency and outcome.

Run:
    uv run python load_test/driver.py --rps 8 --duration 300

Writes a JSON file (default results/load_test.json) with summary + raw
per-request data.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
PERF_POOL = ROOT / "load_test" / "perf_pool.jsonl"
DEFAULT_OUT = ROOT / "results" / "load_test.json"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


async def fire_one(
    session: aiohttp.ClientSession,
    url: str,
    question: dict,
    results: list[dict],
) -> None:
    payload = {"question": question["question"], "db": question["db_id"]}
    t0 = time.monotonic()
    status = "ok"
    err: str | None = None
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            await resp.read()
            if resp.status != 200:
                status = "http_error"
                err = f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        status = "timeout"
    except Exception as e:  # noqa: BLE001
        status = "client_error"
        err = f"{type(e).__name__}: {e}"
    results.append({
        "latency_seconds": time.monotonic() - t0,
        "status": status,
        "error": err,
    })


async def drive(args: argparse.Namespace) -> None:
    if not PERF_POOL.exists():
        raise SystemExit(f"{PERF_POOL} not found - run scripts/load_data.py first")
    questions = [json.loads(line) for line in PERF_POOL.read_text().splitlines() if line.strip()]
    if not questions:
        raise SystemExit(f"{PERF_POOL} is empty")

    rnd = random.Random(0)
    results: list[dict] = []
    interval = 1.0 / args.rps

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.monotonic()
        deadline = start + args.duration
        tasks: list[asyncio.Task] = []
        next_fire = start
        while time.monotonic() < deadline:
            q = rnd.choice(questions)
            tasks.append(asyncio.create_task(fire_one(session, args.agent_url, q, results)))
            next_fire += interval
            sleep_for = next_fire - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        if tasks:
            await asyncio.wait(tasks, timeout=60.0)
        wall = time.monotonic() - start

    latencies = sorted(r["latency_seconds"] for r in results if r["status"] == "ok")

    def pct(p: float) -> float:
        if not latencies:
            return float("nan")
        k = int(round(p * (len(latencies) - 1)))
        return latencies[k]

    summary = {
        "requested_rps": args.rps,
        "duration_seconds": args.duration,
        "wall_clock_seconds": wall,
        "total_requests": len(results),
        "achieved_rps": (len(results) / wall) if wall > 0 else 0.0,
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "timeouts": sum(1 for r in results if r["status"] == "timeout"),
        "http_errors": sum(1 for r in results if r["status"] == "http_error"),
        "client_errors": sum(1 for r in results if r["status"] == "client_error"),
        "latency_p50": pct(0.50),
        "latency_p95": pct(0.95),
        "latency_p99": pct(0.99),
        "latency_max": max(latencies, default=float("nan")),
    }

    output = {"summary": summary, "results": results}

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rps", type=int, default=10)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT, dest="agent_url")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    print(f"Starting load test: {args.rps} RPS for {args.duration}s → {args.agent_url}")
    asyncio.run(drive(args))


if __name__ == "__main__":
    main()
