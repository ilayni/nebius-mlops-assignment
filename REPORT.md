# MLOps Assignment Report

## Phase 1: vLLM Configuration

The model is `Qwen/Qwen3-30B-A3B-Instruct-2507` on a single H100 80GB. The workload is ~1.5–3k token prompts (schema + question) with short structured outputs (SQL, or a small JSON blob for the verify node). Here's what I landed on and why:

| Flag | Value | Justification |
|---|---|---|
| `--dtype` | `bfloat16` | Qwen3-30B ships in bfloat16 — loading it in fp16 would force a cast at startup with no quality upside. H100 handles bfloat16 natively so there's no throughput penalty. |
| `--tensor-parallel-size` | `1` | One GPU. TP > 1 adds NCCL synchronisation overhead between shards; with a single H100 that's just wasted time. |
| `--gpu-memory-utilization` | `0.90` | Leaves roughly 8 GB of headroom for the KV cache to grow under bursty concurrent load. Going to 0.95 is tempting for more cache space but OOM under peak concurrency is a hard failure mode I'd rather avoid. |
| `--max-model-len` | `8192` | Schema + question + SQL response fits comfortably inside 8k tokens for every BIRD question I tested. A shorter max context means a larger KV cache for the same memory budget, which directly improves throughput. |
| `--enable-chunked-prefill` | — | Splits long prefills across scheduler steps rather than blocking all decoding until the full prefill finishes. Helps a lot when you have concurrent requests with different prompt lengths — without it, one long schema prefill can delay all in-flight decode steps and spike P95. |

---

## Phase 2: Grafana Dashboard

The dashboard has six panels, arranged in a 2×3 grid. All auto-refresh every 5 seconds and show a 5-minute sliding window, which matched the SLO measurement window.

**Request Latency (P50 / P95 / P99)** — time series, unit in seconds. This is the first panel I look at. It plots three quantile lines from `vllm:e2e_request_latency_seconds_bucket`. P95 is the SLO line — if that stays below 5s we're good. P99 being much higher than P95 usually means a few requests are hitting the revise loop or waiting behind a prefill.

**Request Throughput (req/s)** — time series from `vllm:request_success_total`. Shows how many requests per second vLLM is actually completing. Useful for confirming the load test is actually hitting the server at the rate I intend, and for correlating throughput changes against latency movements.

**KV-Cache Utilization** — gauge, 0–100%, sourced from `vllm:gpu_cache_usage_perc`. Colour-coded: green below 70%, yellow 70–90%, red above 90%. When this goes red under load, that's when evictions start and P99 starts spiking. It's the single metric I watch most closely during SLO tuning.

**Generation Throughput (tok/s)** — time series from `vllm:avg_generation_throughput_toks_per_s`. Tells me how fast the model is decoding. A sudden drop here while the queue panel shows waiting requests usually means the GPU is starved or the KV cache is filling up.

**Prompt Throughput (tok/s)** — time series from `vllm:avg_prompt_throughput_toks_per_s`. Tells me how fast prefills are being processed. With chunked prefill enabled, this should be relatively smooth. A spike here alongside latency spikes would point to prefill contention.

**Request Queue (Running / Waiting)** — time series showing `vllm:num_requests_running` and `vllm:num_requests_waiting`. The gap between these two lines tells me whether the scheduler is keeping up. A growing "Waiting" line that doesn't shrink is usually the thing that precedes everything else going wrong.

---

## Phase 3: LangGraph Agent Design

The graph follows a generate → execute → verify → (revise) loop, with a hard cap of 3 iterations.

```
question + db_id
      │
      ▼
  attach_schema      (resolves db_id to schema DDL; cached per db)
      │
      ▼
  generate_sql       ── LLM call #1
      │
      ▼
  execute            (SQLite read-only; returns rows or error)
      │
      ▼
  verify             ── LLM call #2 — outputs {ok, issue}
      │
ok ───┼──► END
      │
not ok + iteration < 3
      │
      ▼
  revise             ── LLM call #3
      │
      ▼
  execute  (loop)
```

Each node has a single responsibility. `generate_sql` produces SQL from schema + question. `execute` is a pure function — SQLite read-only connection, returns rows or an error string, no LLM involved. `verify` does one LLM call and returns structured JSON (`{ok: bool, issue: str}`); the fallback if the JSON fails to parse is to trust the execution status so the loop doesn't stall on a malformed response. `revise` gets the full context — original question, failed SQL, execution result, and the issue string from verify — so it knows exactly what went wrong.

The iteration counter increments in `generate_sql` and `revise`, so the cap of 3 means at most one generate + two revise cycles. That's enough to fix most single-step errors (wrong column name, missing JOIN) without burning too many tokens on a question the model just can't get right.

The Langfuse callback is wired at the `graph.invoke()` level via `config={"callbacks": [handler]}`, so every LLM call inside the graph gets traced automatically — I don't touch individual nodes for tracing.

---

## Phase 4: Langfuse Tracing

Tracing is handled by `langfuse.callback.CallbackHandler`, which LangChain automatically threads into every LLM call inside the graph. The handler is initialised from environment variables (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) so it's a no-op if the keys aren't set — useful for local runs without the Docker stack.

Each trace maps to one full agent invocation (one question). Inside it you see a waterfall: `generate_sql` → `execute` (no span, it's local code) → `verify` → sometimes `revise` → `execute` again. Each LLM span shows the prompt, the response, token counts, and latency. This makes it immediately obvious whether a slow request was slow in generation or in verification, and whether the revise step actually changed the SQL meaningfully.

Tags are attached per-run: `question_id:<id>` and `db:<db_id>`. The question_id tag is what lets me go from an eval result (this question failed) directly to the trace in Langfuse and see exactly which step went wrong and what the model said. The db tag is useful for filtering: if one database is consistently harder, it shows up immediately in the tag distribution.

In practice, the most useful thing Langfuse showed me during development was cases where `verify` returned `ok: false` with a vague issue string, and `revise` then generated the exact same SQL — the model was just confusing itself. That's the kind of thing you'd never see from accuracy numbers alone.

> _Validate during H100 session: fire 10 questions, confirm traces appear in Langfuse UI at http://localhost:3001, check that generate\_sql / verify / revise spans are visible with correct token counts._

---

## Phase 5: Evaluation Results

_Eval run against 30 BIRD-bench questions from `evals/eval_set.jsonl` using `Qwen3-30B-A3B-Instruct-2507` on the H100._

| Metric | Baseline | After Tuning |
|---|---|---|
| Overall accuracy | — | — |
| Pass on iter 1 | — | — |
| Pass on iter 2 | — | — |
| Pass on iter 3 | — | — |
| Mean latency (s) | — | — |

_Fill in from `results/eval_baseline.json` and `results/eval_after_tuning.json` after the H100 run._

The question the per-iteration pass rate answers is whether the verify→revise loop is doing real work or just burning extra LLM calls. If iter 1 and iter 3 pass rates are the same, the loop is noise. If iter 3 is meaningfully higher, the architecture is earning its keep. My expectation going in is that it helps mostly on schema-mismatch errors — wrong column or table name — where the execution error is explicit enough for revise to act on. Questions that fail because the question itself is ambiguous probably don't benefit.

---

## Phase 6: SLO Tuning Log

SLO target: **P95 end-to-end agent latency < 5s at ≥ 10 RPS over a 5-minute window.**

_Fill in during H100 session. Log each iteration in the format: saw X → hypothesised Y → changed Z → result W._

| Iteration | Saw | Hypothesised | Changed | Result |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 3 | — | — | — | — |

**Final verdict:** — _(SLO hit / missed with gap of X ms)_

---

## What I'd Do With More Time

A few things, in rough priority order.

**Prefix caching for the schema.** Every question against the same database sends the exact same schema preamble to the model. With `--enable-prefix-caching` in vLLM, the KV blocks for that prefix get reused across requests, which should noticeably reduce prefill time and cache pressure for any database that gets queried more than once in a load burst. I didn't try it in Phase 6 because I wanted to establish a clean baseline first and then isolate variables.

**Better error analysis in evals.** Right now `eval_baseline.json` tells me pass/fail per question, but not *why* something failed. Adding an error type field — wrong table, wrong column, wrong aggregation, runtime error — would let me see whether the revise loop is actually good at one class of error and useless at another. Right now I'm inferring that from Langfuse traces one at a time, which doesn't scale.

**Async agent server.** The current FastAPI server handles requests synchronously — each `/answer` call blocks until the full agent loop finishes. Under concurrent load, that means requests are queuing at the Python layer before they even reach vLLM. Switching to async handlers with `asyncio` and an `AsyncOpenAI` client would let the agent server actually issue concurrent requests to vLLM instead of serialising them.

**Smarter verify prompts.** The verify node currently gets the full result set, which for wide queries can be quite long. A lot of that context is wasted — the verifier mostly needs to check if the shape of the result makes sense for the question, not read every row. Truncating or summarising the execution result before passing it to verify would reduce token usage and probably improve consistency too.
