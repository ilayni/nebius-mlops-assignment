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

Validated during the H100 session: ran the full 30-question eval and confirmed traces appeared in Langfuse at http://localhost:3001 with `generate_sql` / `verify` / `revise` spans visible and token counts logged per span. Screenshots `langfuse_trace.png` and `langfuse_tags.png` capture a multi-iteration trace and the tag distribution across the eval run.

---

## Phase 5: Evaluation Results

_Eval run against 30 BIRD-bench questions from `evals/eval_set.jsonl` using `Qwen3-30B-A3B-Instruct-2507` on the H100._

| Metric | Baseline | After Tuning |
|---|---|---|
| Overall accuracy | 20.0% | 3.3% |
| Pass on iter 1 | 23.3% | 3.3% |
| Pass on iter 2 | 23.3% | 3.3% |
| Pass on iter 3 | 20.0% | 3.3% |
| Mean latency (s) | 10.1s | 37.4s |

The per-iteration pass rates tell an interesting story at baseline: accuracy is flat from iter 1 to iter 2 (23.3% both), then drops to 20.0% at iter 3. The loop is not helping — it is slightly hurting. This matches what I observed in Langfuse traces: `revise` sometimes generates the exact same SQL or drifts to a worse query when the verify failure message is vague. The 3.3 percentage-point drop from iter 1 to iter 3 suggests the model is occasionally revising a correct (or near-correct) query into a wrong one when verify incorrectly flags it. The loop earns its keep on explicit execution errors (wrong column, missing JOIN) but adds noise on schema-ambiguous questions where verify can't give a precise issue string.

The after-tuning run shows a sharp regression: 3.3% accuracy (1/30) vs 20.0% at baseline, and mean latency nearly quadrupled to 37.4s. Breaking the 30 results down by failure mode tells the real story: **1 correct, 22 that executed cleanly but returned the wrong rows, and 7 that failed outright (6 SQL syntax errors + 1 timeout).** So the dominant failure was *semantic* — the model produced runnable-but-incorrect SQL — not the syntax-level `<think>` bleed-through I first suspected: only 2 of the 30 predictions actually contained `<think>` tokens, so that explains a couple of the syntax errors at most, not the collapse. A near-total wipeout like this at temperature 0 points to the final serving config itself being degraded rather than a single prompt quirk. Of the changes layered on during the H100 session (prefix caching, `--max-num-seqs 16`, and a hand-patched tokenizer), the tokenizer `getattr` fallback is the only one that plausibly alters generation — prefix caching and the scheduler cap don't touch decoding — so that patch is the most likely culprit, though I didn't get a clean A/B before the session ended to confirm it. The one 120s timeout also indicates the request queue was under pressure during the run. Either way, the honest read is that quality did **not** survive tuning, and the flat per-iteration rates (3.3% across all three iterations) show the revise loop salvaged none of it.

---

## Phase 6: SLO Tuning Log

SLO target: **P95 end-to-end agent latency < 5s at ≥ 10 RPS over a 5-minute window.**

_Fill in during H100 session. Log each iteration in the format: saw X → hypothesised Y → changed Z → result W._

| Iteration | Saw | Hypothesised | Changed | Result |
|---|---|---|---|---|
| 1 | 10 RPS for 300s: 4/3000 ok, P95=96s, 1926 timeouts, 1070 client errors, achieved RPS=8.3 | Queue depth is the bottleneck — 10 RPS × 10s/req = ~100 concurrent requests, single uvicorn process + vLLM queue saturates immediately | Baseline run, no changes yet | SLO **missed** by 91s (P95 96s vs 5s target) |
| 2 | 4-worker agent + 10 RPS 180s: 52/1800 ok, P95=114s — **invalid run**: started with ~40 leftover requests, KV cache already at 100%, waiting queue oscillated 0→peak throughout | 4 workers made things worse — more parallel agent paths meant more requests hitting an already-saturated vLLM; KV cache full from the start = every new request queueing immediately | Restarted uvicorn with `--workers 4` | SLO **missed**; result not comparable to run 1 due to dirty state; need clean restart |
| 3 | vLLM restarted (clean queue), 2 RPS diagnostic: 24/240 ok, P95=76s, P50=36s, achieved RPS=1.33 | Capacity is well below 2 RPS — each agent request makes 2-3 sequential LLM calls; queue saturates even at low load; problem is fundamental throughput, not queue management | Reduced to --rps 2 to find actual capacity ceiling | SLO **missed** by 71s; system capacity < 2 RPS with current config; next lever: --max-num-seqs |
| 4 | --max-num-seqs 16 + 2 RPS: 6/240 ok, P95=89s, P50=11s — worse than run 3 | Capping scheduler to 16 seqs made vLLM reject most requests fast (121 client errors); P50 improved for the few that got through but throughput collapsed | Added --max-num-seqs 16, restarted vLLM | SLO **missed**; wrong trade-off — kills throughput to lower individual latency; **SLO declared missed** |

**Prefix caching — attempted, patched, and running:** Added `--enable-prefix-caching` to `scripts/start_vllm.sh` to reuse KV blocks for repeated schema prefixes (~1-2k tokens, identical per database). vLLM crashed on startup with `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended` — a version incompatibility between vLLM and transformers 5.10.2, where the newer transformers version dropped an attribute vLLM's tokenizer wrapper expects. A one-line patch (`getattr` with fallback) in the venv's tokenizer.py restored vLLM startup. Prefix caching ran and showed a ~39.6% cache hit rate. However, a clean comparative load test with prefix caching enabled was not completed before the session ended, so its impact on P95 latency under load remains unconfirmed.

**Final verdict:** SLO **missed**. Best result was run 3: P95 = 76s at 2 RPS — gap of 71s vs 5s target, at one-fifth of the required throughput. Root cause: 2-3 sequential LLM calls per agent request (~10s end-to-end at low concurrency) means the system cannot sustain the 10 req/s × <5s P95 target without a fundamental reduction in per-request LLM call count or significantly faster inference (quantisation, speculative decoding, or stronger hardware).

---

## What I'd Do With More Time

A few things, in rough priority order.

**Comparative load test with prefix caching.** Prefix caching was enabled and ran with a ~39.6% cache hit rate, but a clean comparative load test (same RPS/duration as run 3, prefix caching on vs off) was not completed before the session ended. A back-to-back comparison is the missing piece — the hit rate suggests meaningful KV reuse for repeated schema prefixes, but whether it translates to a measurable P95 reduction under load is still an open question.

**Better error analysis in evals.** Right now `eval_baseline.json` tells me pass/fail per question, but not *why* something failed. Adding an error type field — wrong table, wrong column, wrong aggregation, runtime error — would let me see whether the revise loop is actually good at one class of error and useless at another. Right now I'm inferring that from Langfuse traces one at a time, which doesn't scale.

**Async agent server.** The current FastAPI server handles requests synchronously — each `/answer` call blocks until the full agent loop finishes. Under concurrent load, that means requests are queuing at the Python layer before they even reach vLLM. Switching to async handlers with `asyncio` and an `AsyncOpenAI` client would let the agent server actually issue concurrent requests to vLLM instead of serialising them.

**Smarter verify prompts.** The verify node currently gets the full result set, which for wide queries can be quite long. A lot of that context is wasted — the verifier mostly needs to check if the shape of the result makes sense for the question, not read every row. Truncating or summarising the execution result before passing it to verify would reduce token usage and probably improve consistency too.
