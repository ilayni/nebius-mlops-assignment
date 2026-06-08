# Nebius Academy MLOps Assignment — Implementation Plan

## Context

Build a text-to-SQL proof-of-concept with vLLM serving, LangGraph agent (generate_sql → verify → revise loop), Langfuse tracing, Prometheus/Grafana observability, and an evaluation harness over BIRD-bench. Full assignment: https://github.com/GlebBerjoskin/mlops-assignment

**Key constraints:**
- Mac Pro has no CUDA — vLLM cannot run locally
- H100 is in a separate Nebius tenant — no data transfer from other VMs; model weights must download during the H100 session
- H100 window is 8 hours — must be used efficiently
- All screenshots and results JSONs must come from the H100 run

**Local model backend:** Nebius Token Factory (OpenAI-compatible API at `https://api.tokenfactory.nebius.com/v1`). Key stored as `NEBIUS_API_KEY` in `.env`.

---

## What Runs Where

| Work | Where |
|---|---|
| Project scaffold, docker-compose, Grafana JSON | Mac |
| LangGraph agent, prompts, Langfuse tracing | Mac → Nebius token factory API |
| BIRD-bench eval harness, scoring logic | Mac → Nebius token factory API |
| Load generator script | Mac (dry-run) |
| H100 runbook (authored locally, executed on H100) | Mac → H100 |
| vLLM serving + Prometheus scraping | H100 only |
| All screenshots, results JSONs, SLO tuning | H100 only |
| REPORT.md | Mac (draft), finalize after H100 |

---

## Current State (after A1–A5)

### What is done and working
- `infra/docker-compose.yml` — Prometheus + Grafana + Langfuse v2 + Postgres running
- `infra/prometheus/prometheus.yml` — scraping configured
- `infra/grafana/` — dashboard provisioned, panels rendering
- `agent/graph.py` — LangGraph agent (generate_sql → execute → verify → revise) using `openai.OpenAI` directly, TypedDict state, programmatic verify
- `agent/prompts.py` — 4 prompt templates (generate, revise)
- Tests 1/2/4 pass against Nebius Token Factory

### What is wrong / incomplete
- **Langfuse v2 is wrong**: assignment requires Langfuse v3 (ClickHouse + Redis + MinIO stack), SDK `>=4.0,<5.0`, and `langfuse.langchain.CallbackHandler` for automatic tracing. Our manual span approach is incompatible.
- **Agent architecture is wrong**: assignment uses `langchain_openai.ChatOpenAI`, `@dataclass` state, LLM-based verify node, and serves via FastAPI at `localhost:8001/answer`. Eval harness hits HTTP, not Python.
- **Missing files**: `agent/execution.py`, `agent/schema.py`, `agent/server.py`, `scripts/load_data.py` — all required by assignment scaffold.
- **`evals/run_eval.py`** is a stub — needs full `eval_one()` and `summarize()` implementation.
- **`requirements.txt`** — has wrong deps (langgraph 0.2, langfuse 2.x, missing fastapi/uvicorn/httpx/langchain-openai).

---

## Migration Plan (A6 — do this next)

### Step 1 — Tear down Langfuse v2

```bash
cd infra
docker compose down
docker volume rm infra_postgres_data 2>/dev/null || true
```

### Step 2 — Update `infra/docker-compose.yml`

Replace the Langfuse section entirely. Keep Prometheus and Grafana unchanged. New file:

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.51.2
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    extra_hosts:
      - "host.docker.internal:host-gateway"
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
      - --web.enable-lifecycle
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.4.3
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/provisioning/dashboards:/var/lib/grafana/dashboards:ro
    depends_on:
      - prometheus
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 5s
      timeout: 5s
      retries: 10
    restart: unless-stopped

  clickhouse:
    image: clickhouse/clickhouse-server:24.3
    environment:
      CLICKHOUSE_DB: default
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  minio:
    image: minio/minio:RELEASE.2024-06-13T22-53-53Z
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio1234
    volumes:
      - minio_data:/data
    restart: unless-stopped

  minio-init:
    image: minio/mc:latest
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
        sleep 5;
        mc alias set local http://minio:9000 minio minio1234;
        mc mb --ignore-existing local/langfuse;
        exit 0;
      "

  langfuse-web:
    image: langfuse/langfuse:3
    ports:
      - "3001:3000"
    depends_on:
      postgres:
        condition: service_healthy
      clickhouse:
        condition: service_started
      redis:
        condition: service_started
      minio-init:
        condition: service_completed_successfully
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@postgres:5432/langfuse
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      LANGFUSE_S3_MEDIA_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID: minio
      LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY: minio1234
      LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE: "true"
      NEXTAUTH_URL: http://localhost:3001
      NEXTAUTH_SECRET: supersecret-change-me
      SALT: supersalt-change-me
      TELEMETRY_ENABLED: "false"
      LANGFUSE_INIT_ORG_NAME: mlops
      LANGFUSE_INIT_PROJECT_NAME: text-to-sql
      LANGFUSE_INIT_PROJECT_PUBLIC_KEY: pk-lf-11111111-2222-3333-4444-555555555555
      LANGFUSE_INIT_PROJECT_SECRET_KEY: sk-lf-66666666-7777-8888-9999-aaaaaaaaaaaa
      LANGFUSE_INIT_USER_EMAIL: admin@local.dev
      LANGFUSE_INIT_USER_PASSWORD: admin123
      LANGFUSE_INIT_USER_NAME: Admin
    restart: unless-stopped

  langfuse-worker:
    image: langfuse/langfuse-worker:3
    depends_on:
      postgres:
        condition: service_healthy
      clickhouse:
        condition: service_started
      redis:
        condition: service_started
      minio-init:
        condition: service_completed_successfully
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@postgres:5432/langfuse
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      LANGFUSE_S3_MEDIA_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID: minio
      LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY: minio1234
      LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE: "true"
      NEXTAUTH_SECRET: supersecret-change-me
      SALT: supersalt-change-me
      TELEMETRY_ENABLED: "false"
    restart: unless-stopped

volumes:
  postgres_data:
  clickhouse_data:
  minio_data:
```

Bring up new stack:
```bash
docker compose up -d
# Wait ~60s for postgres healthcheck, then verify:
curl -s http://localhost:3001/api/public/health   # should return {"status":"ok"}
```

Login: `admin@local.dev` / `admin123`. Keys are pre-seeded (no manual API key creation needed).

### Step 3 — Update `.env`

Replace dead v2 keys with v3 pre-seeded keys. Add `VLLM_BASE_URL` alias. Full `.env`:

```
OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
VLLM_BASE_URL=https://api.tokenfactory.nebius.com/v1
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
NEBIUS_API_KEY=<existing value — do not change>
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-11111111-2222-3333-4444-555555555555
LANGFUSE_SECRET_KEY=sk-lf-66666666-7777-8888-9999-aaaaaaaaaaaa
```

### Step 4 — Update `requirements.txt`

```
langgraph>=1.0,<2.0
langchain>=0.3,<1.0
langchain-openai>=0.3,<1.0
langfuse>=4.0,<5.0
fastapi>=0.115,<1.0
uvicorn[standard]>=0.30,<1.0
pydantic>=2.0,<3.0
httpx>=0.27,<1.0
python-dotenv>=1.0,<2.0
datasets>=2.20
tqdm>=4.66,<5.0
```

Recreate venv:
```bash
cd /Users/YOUR_USER/dev/cloud/nebius/lesson17_observabiility_evaluation_mlops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 5 — Create `agent/execution.py`

```python
from dataclasses import dataclass, field
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    row_count: int = 0

    def render(self, max_rows: int = 10) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        if self.row_count == 0:
            return "OK: 0 rows returned."
        cols = ", ".join(self.columns or [])
        preview = "\n".join(
            " | ".join(str(c) for c in row) for row in (self.rows or [])[:max_rows]
        )
        more = (
            f"\n... ({self.row_count - max_rows} more rows)"
            if self.row_count > max_rows
            else ""
        )
        return f"OK: {self.row_count} rows.\nCOLUMNS: {cols}\nFIRST ROWS:\n{preview}{more}"


def execute_sql(db_id: str, sql: str, timeout_seconds: float = 5.0) -> ExecutionResult:
    from agent.schema import db_path
    path = db_path(db_id)
    try:
        with sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, timeout=timeout_seconds
        ) as conn:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return ExecutionResult(ok=True, rows=rows, columns=cols, row_count=len(rows))
    except Exception as e:
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")
```

### Step 6 — Create `agent/schema.py`

```python
from functools import lru_cache
from pathlib import Path
import sqlite3

ROOT = Path(__file__).parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def available_dbs() -> list[str]:
    return [p.stem for p in DB_DIR.glob("*.sqlite")]


@lru_cache(maxsize=256)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        )
        tables = [row[0] for row in cur.fetchall()]
        # foreign keys
        fks = []
        for table_sql in tables:
            name = table_sql.split("(")[0].split()[-1].strip('"')
            cur.execute(f"PRAGMA foreign_key_list(\"{name}\")")
            for row in cur.fetchall():
                fks.append(
                    f"-- FK: {name}.{row[3]} -> {row[2]}.{row[4]}"
                )
    schema = "\n\n".join(tables)
    if fks:
        schema += "\n\n" + "\n".join(fks)
    return schema
```

### Step 7 — Create `agent/server.py`

```python
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, Any] = {}


@app.post("/answer")
def answer(req: AnswerRequest):
    from agent.graph import run
    result = run(
        question=req.question,
        db_id=req.db,
        tags=req.tags,
    )
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
```

Run with: `uvicorn agent.server:app --host 0.0.0.0 --port 8001`

### Step 8 — Rewrite `agent/graph.py`

The new agent uses LangChain's `ChatOpenAI`, `@dataclass` state, and Langfuse v3 `CallbackHandler`.

Key differences from our A4/A5 implementation:
- State is `@dataclass` not `TypedDict`
- LLM is `ChatOpenAI` from `langchain_openai` (supports LangChain callbacks)
- Verify is an LLM call that returns JSON `{"ok": true/false, "issue": "..."}`
- Tracing is automatic via `langfuse.langchain.CallbackHandler` passed to `llm.invoke(config={"callbacks": [handler]})`
- Entry point is `run(question, db_id, tags)` not `run(question, db_path, question_id, db_name)`

```python
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agent import prompts
from agent.execution import execute_sql, ExecutionResult
from agent.schema import render_schema

load_dotenv()

MAX_ITERATIONS = 3

LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("VLLM_BASE_URL")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("NEBIUS_API_KEY", "not-needed")
LLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct")


def llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0,
        max_tokens=512,
    )


# Langfuse v3 callback handler — silently disabled when keys are absent
def _langfuse_handler():
    try:
        from langfuse.langchain import CallbackHandler
        if os.environ.get("LANGFUSE_PUBLIC_KEY"):
            return CallbackHandler(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                host=os.environ.get("LANGFUSE_HOST", "http://localhost:3001"),
            )
    except Exception:
        pass
    return None


def _extract_sql(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _parse_verify(text: str, execution: ExecutionResult) -> tuple[bool, str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        ok_val = data.get("ok", True)
        if isinstance(ok_val, str):
            ok_val = ok_val.lower() == "true"
        return bool(ok_val), str(data.get("issue", ""))
    except (json.JSONDecodeError, AttributeError):
        return execution.ok, "Could not parse verifier response"


@dataclass
class AgentState:
    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: Optional[ExecutionResult] = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict] = field(default_factory=list)
    callbacks: list = field(default_factory=list)


def generate_sql_node(state: AgentState) -> dict:
    response = llm().invoke(
        [
            ("system", prompts.GENERATE_SQL_SYSTEM),
            ("human", prompts.GENERATE_SQL_USER.format(
                schema=state.schema, question=state.question
            )),
        ],
        config={"callbacks": state.callbacks},
    )
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_sql_node(state: AgentState) -> dict:
    result = execute_sql(state.db_id, state.sql)
    return {"execution": result}


def verify_node(state: AgentState) -> dict:
    response = llm().invoke(
        [
            ("system", prompts.VERIFY_SYSTEM),
            ("human", prompts.VERIFY_USER.format(
                question=state.question,
                sql=state.sql,
                execution=state.execution.render() if state.execution else "No result",
            )),
        ],
        config={"callbacks": state.callbacks},
    )
    ok, issue = _parse_verify(response.content, state.execution)
    return {"verify_ok": ok, "verify_issue": issue}


def revise_node(state: AgentState) -> dict:
    response = llm().invoke(
        [
            ("system", prompts.REVISE_SYSTEM),
            ("human", prompts.REVISE_USER.format(
                schema=state.schema,
                question=state.question,
                sql=state.sql,
                execution=state.execution.render() if state.execution else "No result",
                issue=state.verify_issue,
            )),
        ],
        config={"callbacks": state.callbacks},
    )
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_verify(state: AgentState) -> str:
    if state.verify_ok:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


_workflow = StateGraph(AgentState)
_workflow.add_node("generate_sql", generate_sql_node)
_workflow.add_node("execute_sql", execute_sql_node)
_workflow.add_node("verify", verify_node)
_workflow.add_node("revise", revise_node)
_workflow.set_entry_point("generate_sql")
_workflow.add_edge("generate_sql", "execute_sql")
_workflow.add_edge("execute_sql", "verify")
_workflow.add_conditional_edges(
    "verify", route_after_verify, {"revise": "revise", "end": END}
)
_workflow.add_edge("revise", "execute_sql")
_graph = _workflow.compile()


def run(question: str, db_id: str, tags: dict = None) -> dict:
    schema = render_schema(db_id)
    handler = _langfuse_handler()
    callbacks = [handler] if handler else []

    t0 = time.monotonic()
    final = _graph.invoke(
        AgentState(
            question=question,
            db_id=db_id,
            schema=schema,
            callbacks=callbacks,
        )
    )
    latency = round(time.monotonic() - t0, 3)

    exec_result: Optional[ExecutionResult] = final.get("execution")
    return {
        "sql": final.get("sql", ""),
        "rows": exec_result.rows if exec_result and exec_result.ok else None,
        "iterations": final.get("iteration", 0),
        "ok": final.get("verify_ok", False),
        "error": exec_result.error if exec_result else None,
        "history": final.get("history", []),
        "latency_s": latency,
    }
```

### Step 9 — Rewrite `agent/prompts.py`

```python
GENERATE_SQL_SYSTEM = (
    "You are a SQL expert. Write a single SQLite SQL query that answers the question. "
    "Return ONLY the SQL — no explanation, no markdown."
)

GENERATE_SQL_USER = "Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"

VERIFY_SYSTEM = (
    "You are a SQL result checker. Decide if the execution result plausibly answers the question. "
    'Respond ONLY with JSON: {"ok": true, "issue": ""} or {"ok": false, "issue": "reason"}.'
)

VERIFY_USER = "Question: {question}\n\nSQL: {sql}\n\nResult: {execution}\n\nJSON:"

REVISE_SYSTEM = (
    "You are a SQL expert. Fix the SQL query that failed verification. "
    "Return ONLY the corrected SQL — no explanation, no markdown."
)

REVISE_USER = (
    "Schema:\n{schema}\n\nQuestion: {question}\n\n"
    "Failed SQL:\n{sql}\n\nExecution:\n{execution}\n\nIssue: {issue}\n\nCorrected SQL:"
)
```

### Step 10 — Create `scripts/load_data.py`

Downloads BIRD dev set, creates `data/bird/<db_id>.sqlite`, and writes eval sets.

```python
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
    # Normalize to common schema: question, db_id, gold_sql
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
```

### Step 11 — Implement `evals/run_eval.py`

```python
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
    per_iter = {}
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
        per_iter[f"iter_{k}"] = round(correct / n, 3)

    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    return {
        "n": n,
        "accuracy": round(overall, 3),
        "per_iteration_accuracy": per_iter,
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
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
    results = []
    for i, q in enumerate(questions):
        r = eval_one(q, args.agent_url, dry_run=args.dry_run)
        status = "✓" if r["match"] else "✗"
        print(f"  [{i+1:02d}/{len(questions)}] {status} {r['db_id']} — {r['iterations']} iter, {r['latency_s']}s")
        results.append(r)

    summary = summarize(results)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
```

### Step 12 — Create `load_test/` directory + `scripts/load_gen.py`

```bash
mkdir -p load_test
```

`scripts/load_gen.py` — concurrent load generator hitting the agent endpoint:

```python
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
```

### Step 13 — Verify the full stack locally

```bash
# 1. Confirm Langfuse v3 is up
curl -s http://localhost:3001/api/public/health  # {"status":"ok"}

# 2. Start agent server
source .venv/bin/activate
uvicorn agent.server:app --host 0.0.0.0 --port 8001 &

# 3. Smoke test the agent
curl -s -X POST http://localhost:8001/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many rows are in any table?","db":"california_schools"}' | python3 -m json.tool

# 4. Check Langfuse trace appeared
# Open http://localhost:3001 → Traces → should see a trace

# 5. Run eval dry-run (no LLM calls, just verify scoring works)
python evals/run_eval.py --dry-run

# 6. Optionally run real eval (takes ~5-10 min with Token Factory)
python evals/run_eval.py --output results/eval_baseline_local.json
```

---

## Phase A — Local Mac Development (complete before booking H100)

### A3 — Grafana Dashboard JSON (verify still works after migration)

Panels auto-provisioned. After `docker compose up`, open Grafana at `:3000` and confirm panels load. Start mock metrics server if needed:
```bash
python scripts/mock_metrics.py
```

### A7 — Load Generator

After the migration steps above, test locally:
```bash
python scripts/load_gen.py --rps 2 --duration 10 --url http://localhost:8001/answer
```
Confirm it exits cleanly (results or "No results" printed).

### A8 — Screenshot Checklist

`screenshots/CHECKLIST.md` — list all 11 required filenames from the assignment.

### A9 — Deployment to H100

Push to private fork, then SSH to H100:
```bash
# On Mac: push to your private fork
git push origin main

# On H100 (in runbook)
git clone <your-repo> app && cd app
cp .env.example .env
# Edit .env: OPENAI_BASE_URL=http://localhost:8000/v1, OPENAI_API_KEY=EMPTY
```

### A11 — H100 Runbook

`scripts/h100_runbook.sh` — exact ordered shell commands for the entire H100 session.

---

## Phase B — H100 Session (8 hours)

**Start vLLM in the first 5 minutes** — 35-min warmup is the bottleneck.

### 0:00–0:05 — SSH + Immediate vLLM Start

```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 \
    -L 3001:localhost:3001 -L 8000:localhost:8000 -L 8001:localhost:8001 user@H100_IP

git clone <your-repo> app && cd app

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000 &
```

### 0:05–0:40 — Install + Start Stack (while vLLM warms up)

```bash
pip install -r requirements.txt
docker compose -f infra/docker-compose.yml up -d
# Edit .env: OPENAI_BASE_URL=http://localhost:8000/v1, NEBIUS_API_KEY=EMPTY
uvicorn agent.server:app --host 0.0.0.0 --port 8001 &
```

### 0:40–1:00 — Validate

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/metrics | grep vllm | head -20
curl http://localhost:8001/health
```

Screenshot: **`vllm_manual_query.png`**

### 1:30–2:00 — Phase 1: vLLM Flag Justification

Document each flag in REPORT.md.

### 2:00–2:30 — Baseline Eval

```bash
python evals/run_eval.py --output results/eval_baseline.json
```

Screenshots: **`grafana_eval_run.png`**, **`langfuse_trace.png`**, **`langfuse_tags.png`**

### 2:30–7:00 — SLO Tuning

SLO: P95 < 5s at 10+ RPS over 5-min window.

```bash
python scripts/load_gen.py --rps 12 --duration 300 --url http://localhost:8001/answer
```

Screenshots: **`grafana_before.png`**, **`grafana_after.png`**, **`grafana_serving.png`**

After best config:
```bash
python evals/run_eval.py --output results/eval_after_tuning.json
```

### 7:00–8:00 — Screenshots + Report + Commit

---

## Submission Checklist

### Code
- [ ] `agent/execution.py`
- [ ] `agent/schema.py`
- [ ] `agent/server.py`
- [ ] `agent/graph.py`
- [ ] `agent/prompts.py`
- [ ] `evals/run_eval.py`
- [ ] `scripts/load_data.py`
- [ ] `scripts/load_gen.py`
- [ ] `infra/docker-compose.yml` (Langfuse v3)
- [ ] `infra/grafana/provisioning/dashboards/serving.json`
- [ ] `results/eval_baseline.json` (from H100)
- [ ] `results/eval_after_tuning.json` (from H100)
- [ ] `REPORT.md`
- [ ] `.env.example`

### Screenshots (7 required)
- [ ] `screenshots/vllm_manual_query.png`
- [ ] `screenshots/grafana_serving.png`
- [ ] `screenshots/langfuse_trace.png`
- [ ] `screenshots/langfuse_tags.png`
- [ ] `screenshots/grafana_eval_run.png`
- [ ] `screenshots/grafana_before.png`
- [ ] `screenshots/grafana_after.png`
