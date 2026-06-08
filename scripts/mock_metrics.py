#!/usr/bin/env python3
"""
Fake vLLM Prometheus metrics server for local Grafana validation.
Emits the same metric names that vLLM uses so all dashboard panels render.

Counters grow linearly with elapsed time so rate() returns non-zero values
and histogram_quantile() resolves to real quantile estimates.

Usage:
    python scripts/mock_metrics.py          # listens on :8000
    python scripts/mock_metrics.py --port 8001
"""
import argparse
import math
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

START = time.time()

LATENCY_BUCKETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
# Cumulative CDF fractions for each bucket (must end with 1.0 for +Inf)
LATENCY_CDF = [0.02, 0.08, 0.20, 0.50, 0.80, 0.95, 0.99]

RATE = 3.0  # simulated requests/second


def _metrics() -> str:
    elapsed = max(1.0, time.time() - START)
    total = int(elapsed * RATE)

    lines = []

    # --- histogram: e2e request latency ---
    lines.append("# HELP vllm:e2e_request_latency_seconds End-to-end request latency")
    lines.append("# TYPE vllm:e2e_request_latency_seconds histogram")
    for le, frac in zip(LATENCY_BUCKETS, LATENCY_CDF):
        lines.append(
            f'vllm:e2e_request_latency_seconds_bucket{{le="{le}"}} {int(total * frac)}'
        )
    lines.append(f'vllm:e2e_request_latency_seconds_bucket{{le="+Inf"}} {total}')
    lines.append(f"vllm:e2e_request_latency_seconds_sum {total * 1.8:.1f}")
    lines.append(f"vllm:e2e_request_latency_seconds_count {total}")

    # --- counter: successful requests ---
    lines.append("# HELP vllm:request_success_total Total successful requests")
    lines.append("# TYPE vllm:request_success_total counter")
    lines.append(f"vllm:request_success_total {total}")

    # --- gauges ---
    lines.append("# HELP vllm:gpu_cache_usage_perc KV-cache utilization (0-1)")
    lines.append("# TYPE vllm:gpu_cache_usage_perc gauge")
    kv = 0.45 + 0.25 * math.sin(elapsed / 30)
    lines.append(f"vllm:gpu_cache_usage_perc {kv:.4f}")

    lines.append("# HELP vllm:avg_generation_throughput_toks_per_s Generation tok/s")
    lines.append("# TYPE vllm:avg_generation_throughput_toks_per_s gauge")
    gen_tps = 820 + 60 * math.sin(elapsed / 20)
    lines.append(f"vllm:avg_generation_throughput_toks_per_s {gen_tps:.1f}")

    lines.append("# HELP vllm:avg_prompt_throughput_toks_per_s Prompt tok/s")
    lines.append("# TYPE vllm:avg_prompt_throughput_toks_per_s gauge")
    prompt_tps = 1200 + 80 * math.cos(elapsed / 25)
    lines.append(f"vllm:avg_prompt_throughput_toks_per_s {prompt_tps:.1f}")

    lines.append("# HELP vllm:num_requests_running Requests currently being processed")
    lines.append("# TYPE vllm:num_requests_running gauge")
    running = max(0, int(3 + 2 * math.sin(elapsed / 10)))
    lines.append(f"vllm:num_requests_running {running}")

    lines.append("# HELP vllm:num_requests_waiting Requests waiting in queue")
    lines.append("# TYPE vllm:num_requests_waiting gauge")
    waiting = max(0, int(1 + math.sin(elapsed / 15)))
    lines.append(f"vllm:num_requests_waiting {waiting}")

    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = _metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress per-request stdout noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Mock vLLM metrics → http://localhost:{args.port}/metrics  (Ctrl-C to stop)")
    HTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
