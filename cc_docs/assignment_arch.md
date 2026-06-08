# Observability Architecture

## Metrics Flow

Prometheus **pulls** (scrapes) — it sends an HTTP GET to the metrics endpoint every 5 seconds.
Grafana queries Prometheus on page load and on each dashboard refresh interval.
Nothing pushes anything.

```
mock_metrics.py        Prometheus              Grafana
(passive HTTP server)  GET /metrics every 5s   query on page load / refresh
       <────────────────────                   <────────────────────
```

On H100, `mock_metrics.py` is replaced by the real vLLM server, which exposes the same
`/metrics` endpoint in Prometheus text format. The pull model is identical.

## Tracing Flow

Langfuse uses a **push** model — the agent sends trace data to Langfuse after each LLM call.

```
LangGraph agent
  (each LLM call via CallbackHandler)
        |
        | POST /api/public/ingestion
        v
  Langfuse web  :3001
        |
        ├──→  Postgres  :5432   (metadata: projects, users, API keys, dataset runs)
        ├──→  ClickHouse         (trace events, spans, observations — the bulk of the data)
        ├──→  Redis               (queue: web → worker async handoff)
        └──→  MinIO               (blob storage: large inputs/outputs, media attachments)

  Langfuse worker (separate process)
        |
        | reads from Redis queue
        v
  writes to ClickHouse + MinIO
```

## Full Stack

```
vLLM / mock_metrics.py  :8000/metrics
        ^
        | GET every 5s
        |
   Prometheus  :9090
        ^
        | PromQL queries
        |
    Grafana  :3000


LangGraph agent (via FastAPI server :8001)
        |
        | POST traces (CallbackHandler)
        v
   Langfuse  :3001
        |
        ├── Postgres  :5432
        ├── ClickHouse
        ├── Redis
        └── MinIO
```

---

## Why Each Component Exists

### Why ClickHouse (not just Postgres for everything)?

Postgres is a row-store — it's fast for small lookups by primary key (find this user, this
project, this API key). It's slow at analytical queries over millions of rows, like:
"show me all spans for all traces in the last 24 hours, grouped by model, sorted by latency."

ClickHouse is a columnar store built specifically for that kind of query. Langfuse stores
every span and observation as an event — at scale this is millions of rows per day. Keeping
that in Postgres would make the UI unusable. So the architecture splits it:

- **Postgres** = relational, transactional metadata (user accounts, API keys, project config,
  dataset definitions). Small tables, row lookups, foreign keys.
- **ClickHouse** = append-only event log (every trace, span, generation, score). Large, write-heavy,
  query-heavy across time ranges and filters.

For this assignment the volume is tiny and either would work, but Langfuse v3 dropped Postgres-only
support and hardcoded ClickHouse for all observations. There's no config switch.

### Why Redis?

The Langfuse web server handles HTTP requests and needs to respond quickly. If it wrote directly
to ClickHouse on every ingestion request, slow ClickHouse writes would block the HTTP response
and create backpressure under load.

Redis is used as a **job queue** between the web process and the worker process:
- Web server receives POST `/api/public/ingestion` → validates → pushes event to Redis queue → returns 200 immediately
- Worker process polls Redis → batches events → writes to ClickHouse + MinIO

This decouples ingestion throughput from write throughput. The agent doesn't wait for
ClickHouse — it gets a fast ack and the worker handles persistence asynchronously.

### Why MinIO?

LLM inputs and outputs can be large — a full schema + question prompt can be several KB, and
Langfuse stores the full input/output of every LLM call. Storing large blobs in Postgres or
ClickHouse is inefficient (row bloat, slow retrieval, no streaming).

MinIO is an S3-compatible object store. Langfuse offloads large payloads (prompt text, completion
text, uploaded media) to MinIO and stores only a reference in ClickHouse. This keeps the columnar
store lean and fast.

In production this would be AWS S3 or GCS. MinIO is the self-hosted equivalent — same API,
runs in Docker, no external dependency.

### Why `langfuse.langchain.CallbackHandler` (not manual spans)?

**The old approach (A5 first attempt):** manually create a `Langfuse` client, call
`trace = langfuse.trace(...)`, then `span = trace.span(...)` around each node, then
`span.end()`. Lots of boilerplate; you have to pass the trace/span objects through the graph
state; easy to forget `.end()` and get unclosed spans.

**The new approach:** LangChain has a callbacks system — when you do
`llm.invoke(..., config={"callbacks": [handler]})`, LangChain fires events
(on_llm_start, on_llm_end, on_chain_start, etc.) and the handler receives them.
`langfuse.langchain.CallbackHandler` listens to those events and automatically:
- Creates a trace per invocation
- Creates a generation span per LLM call, capturing model name, prompt, completion, token counts
- Records latency without any manual `.end()` calls
- Attaches all of it to one trace in the Langfuse UI

This is less code, harder to get wrong, and captures token counts automatically (you'd have to
extract those manually from the response in the old approach). It also works with LangGraph
because LangGraph is built on LangChain — the callbacks propagate through the graph nodes.

The handler is created once per `run()` call and passed into every `llm.invoke()`:

```python
handler = CallbackHandler(public_key=..., secret_key=..., host=...)
llm.invoke([...], config={"callbacks": [handler]})
```

### Why FastAPI server (`agent/server.py`)?

The eval harness (`evals/run_eval.py`) and the load generator (`scripts/load_gen.py`) both
need to send HTTP requests to the agent — they use `httpx.post` and `aiohttp.post` respectively.

Without a server, you'd have to call the Python `run()` function directly, which means:
- The eval script and agent must run in the same process (no isolation)
- The load generator can't do concurrent requests across processes
- On H100, you can't separate the agent process from the eval process

The FastAPI server wraps the `run()` function behind a single POST `/answer` endpoint:

```
POST /answer  {"question": "...", "db": "..."}
→ calls agent.graph.run()
→ returns {"sql": "...", "ok": true, "iterations": 1, ...}
```

This makes the agent a proper service — the eval harness, load generator, and any manual curl
tests all speak HTTP to the same endpoint, exactly as they would in production.

---

## Langfuse v3 Init — Required Environment Variables

Langfuse v3 changed how the initialization seed works. All five `LANGFUSE_INIT_*` variables
are silently ignored unless you also set explicit IDs:

```yaml
LANGFUSE_INIT_ORG_ID: org-mlops-local-00000001   # required — without this, all INIT_* vars are ignored
LANGFUSE_INIT_ORG_NAME: mlops
LANGFUSE_INIT_PROJECT_ID: proj-texttosql-0000001  # required — without this, API keys are not created
LANGFUSE_INIT_PROJECT_NAME: text-to-sql
LANGFUSE_INIT_PROJECT_PUBLIC_KEY: pk-lf-11111111-2222-3333-4444-555555555555
LANGFUSE_INIT_PROJECT_SECRET_KEY: sk-lf-66666666-7777-8888-9999-aaaaaaaaaaaa
LANGFUSE_INIT_USER_EMAIL: admin@local.dev
LANGFUSE_INIT_USER_PASSWORD: admin123
LANGFUSE_INIT_USER_NAME: Admin
```

The IDs are arbitrary strings — they just need to be set so Langfuse knows where to attach
the project and API keys. Using `docker compose restart` does NOT apply new env vars; you
must use `docker compose up -d --force-recreate langfuse-web`.

---

## A4 & A5 Tasks

### What the agent does

Takes a plain English question and a database schema, asks the LLM to write a SQL query,
runs it against a real SQLite database, and returns the result.

```
"How many orders were shipped in January?"
        ↓
  LLM writes SQL
        ↓
  runs against orders.db
        ↓
  [{count: 42}]
```

### What revise does

If the SQL fails to execute (wrong column name, bad syntax, referencing a table that doesn't
exist), the agent sends the error back to the LLM — "you wrote this, it failed with this error,
fix it" — then tries again. It retries up to 3 times. If still failing after 3 revisions it
gives up and returns `ok=false`.

```
LLM writes:  SELECT customer_name FROM orders   ← "customer_name" doesn't exist
sqlite says: no such column: customer_name
revise:      here's the schema, here's what failed, try again
LLM writes:  SELECT status FROM orders          ← correct this time
```

### Full graph flow

```
question + schema
        ↓
  generate_sql  (LLM call)
        ↓
  execute_sql   (sqlite)
        ↓
    verify ──── ok=true ──────────────────────────→ END
        │
        └─── ok=false + iterations < 3 ──→ revise (LLM call)
                                                ↓
                                          execute_sql
                                                ↓
                                             verify  ...repeat...
        └─── ok=false + iterations >= 3 ───────────────────────→ END
```

### Langfuse tracing (A5)

Every LLM call is automatically recorded via `CallbackHandler`. In the Langfuse UI you see
the full chain per question:
- Which SQL the LLM generated
- Which sqlite error came back
- How many tokens each LLM call used
- How many revise iterations it took

This is what the assignment screenshots (`langfuse_trace.png`, `langfuse_tags.png`) need to show.

### Tests

| Test | What it checks |
|---|---|
| 1 — happy path | Agent returns valid SQL, `ok=true`, `iterations=0` — basic flow works |
| 2 — revise trigger | A bad column name forces sqlite to throw; agent retries — revise loop fires at least once |
| 3 — Langfuse trace | After tests 1+2, open localhost:3001 and confirm spans + token counts are visible |
| 4 — iteration cap | Pathological query keeps failing — agent stops cleanly at `MAX_ITERATIONS=3`, no crash |

---

## How the Eval Works

### Ground truth — where does it come from?

The eval set is sourced from **BIRD-bench** (Big Bench for Large-scale Database Grounded Text-to-SQL),
a public academic benchmark. Each entry in the benchmark contains:
- A plain-English question (`"How many employees earn more than 80000?"`)
- The database it refers to (`db_id: "employees"`)
- A human-written SQL query that correctly answers that question (`gold_sql`)

`scripts/load_data.py` downloads the BIRD dev set (~500MB zip), unpacks it, copies each
database's `.sqlite` file into `data/bird/`, and writes a random sample of 30 questions to
`evals/eval_set.jsonl` (seed=0 for reproducibility). The gold SQL in that file is what came
from the benchmark — it was written by humans and verified to produce the correct answer.

### How accuracy is measured — execution accuracy

The eval does **not** compare SQL strings. It compares query results. This is the standard
BIRD-bench metric called *execution accuracy*:

```
gold_sql  → run against the real SQLite DB → gold_rows
pred_sql  → run against the real SQLite DB → pred_rows

match = (canonicalize(gold_rows) == canonicalize(pred_rows))
```

`canonicalize()` converts every value to a string and sorts the rows, so column order and
result order don't affect the score. A question is marked correct if and only if the predicted
SQL produces the same rows as the gold SQL when both are executed against the same database.

This means:
- Two syntactically different SQL queries can both be "correct" (the model doesn't need to
  write the same SQL as the human, just get the same answer)
- A SQL that executes but returns wrong rows scores as incorrect
- A SQL that fails to execute at all scores as incorrect

### What `run_eval.py` does step by step

```
for each question in eval_set.jsonl:
    1. POST {"question": q, "db": db_id} → agent server :8001
       → agent runs generate_sql → execute_sql → verify → (revise?) → returns predicted_sql

    2. Execute gold_sql against data/bird/<db_id>.sqlite → gold_rows
    3. Execute predicted_sql against data/bird/<db_id>.sqlite → pred_rows
    4. match = (canonicalize(gold_rows) == canonicalize(pred_rows))

    5. Also record per_iter_match: for each SQL in the history (first attempt, each revision),
       run it and check if it already matched gold — shows whether the model got it right
       immediately or only after being corrected by the revise loop
```

The final summary reports overall accuracy (fraction of questions where `match=True`) and
per-iteration accuracy (what fraction would have been correct at each revision step).

### Why execute both SQLs instead of just comparing to a stored answer?

Storing pre-computed gold answers would fail for queries that produce different results
depending on the database state (e.g. aggregates, counts). By executing both SQLs live against
the same database file, the eval stays correct even if the database content changes, and it
naturally handles queries where the "correct" answer is an empty result set.

---

## A6 — Local Eval Validation (complete)

Ran `evals/run_eval.py` against the Nebius Token Factory (`Qwen/Qwen3-30B-A3B-Instruct-2507`)
to confirm the full scoring pipeline works end-to-end before the H100 session.

**Results** (`results/eval_a6_nebius.json`):
- 3/3 correct (accuracy 1.0), all solved on first iteration — no revisions triggered
- Mean latency: ~1.03s, well under the 5s SLO target

**What this validated:**
- `predicted_sql` populates correctly from the HTTP response
- The `match=True` path in `eval_one()` executes correctly — the prior `eval_baseline.json` was a dry-run with empty results, so this was the first real execution of the match/summarize logic
- `per_iter_match` fills correctly — `[true]` for each question
- Model string in `.env` (`Qwen/Qwen3-30B-A3B-Instruct-2507`) is confirmed live on Token Factory

**Scoring note — ORDER BY:** `canonicalize()` sorts both result sets before comparing, so a
predicted SQL that omits `ORDER BY` still scores `match=True` if the rows are otherwise equal.
This is correct execution-accuracy semantics (ORDER BY doesn't change what data is returned),
but means ORDER BY correctness is not penalized. Worth knowing when reading accuracy numbers
on the 30-question BIRD set.
