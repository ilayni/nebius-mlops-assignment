# MLOps Assignment — Report

## Phase 1: vLLM Configuration

_Fill in during H100 session._

| Flag | Value | Justification |
|---|---|---|
| `--dtype` | `bfloat16` | |
| `--tensor-parallel-size` | `1` | |
| `--gpu-memory-utilization` | `0.90` | |
| `--max-model-len` | `8192` | |
| `--enable-chunked-prefill` | — | |

## Phase 2: Grafana Dashboard

_Describe each panel and what it measures._

## Phase 3: LangGraph Agent Design

_Agent workflow, node responsibilities, iteration cap._

## Phase 4: Langfuse Tracing

_What is traced, which tags are used, how traces help debugging._

## Phase 5: Evaluation Results

| Metric | Baseline | After Tuning |
|---|---|---|
| Overall accuracy | | |
| Pass on iter 1 | | |
| Pass on iter 2 | | |
| Pass on iter 3+ | | |
| Mean latency (s) | | |

_Honest assessment of whether the revise loop helps._

## Phase 6: SLO Tuning Log

SLO target: P95 < 5s at 10+ RPS sustained over 5 min.

| Iteration | Saw | Hypothesized | Changed | Result |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## What I'd Do With More Time

_Specific, not generic._
