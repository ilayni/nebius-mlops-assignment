"""
Load generator for the text-to-SQL agent.

Usage:
    python scripts/load_gen.py --rps 10 --duration 60 --url http://localhost:8001/answer
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).parent.parent
PERF_POOL = ROOT / "load_test" / "perf_pool.jsonl"


def load_pool() -> list[dict]:
    with open(PERF_POOL) as f:
        return [json.loads(line) for line in f if line.strip()]


async def send_one(session: aiohttp.ClientSession, url: str, question: dict) -> float:
    t0 = time.monotonic()
    try:
        async with session.post(
            url,
            json={"question": question["question"], "db": question["db_id"]},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            await resp.json()
    except Exception:
        pass
    return time.monotonic() - t0


async def run(rps: int, duration: int, url: str):
    pool = load_pool()
    latencies = []
    interval = 1.0 / rps
    t_start = time.monotonic()
    tasks = []

    async with aiohttp.ClientSession() as session:
        while time.monotonic() - t_start < duration:
            q = random.choice(pool)
            tasks.append(asyncio.create_task(send_one(session, url, q)))
            await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, float):
                latencies.append(r)

    if latencies:
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        actual_rps = len(latencies) / duration
        print(f"Requests: {len(latencies)}, RPS: {actual_rps:.1f}, P50: {p50:.2f}s, P95: {p95:.2f}s")
    else:
        print("No results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rps", type=int, default=2)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--url", default="http://localhost:8001/answer")
    args = parser.parse_args()
    asyncio.run(run(args.rps, args.duration, args.url))


if __name__ == "__main__":
    main()
