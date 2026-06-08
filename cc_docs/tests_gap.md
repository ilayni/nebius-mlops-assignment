# Test Gap Analysis

Run date: 2026-06-08  
Command: `python tests/test_agent.py`

---

## Result: FAIL (import error — nothing runs)

```
ImportError: cannot import name 'run' from 'agent.graph'
```

All three tests abort at import time.

---

## Gap 1 — `run()` function missing from `agent/graph.py` (BLOCKER)

**What the test expects:**
```python
from agent.graph import MAX_ITERATIONS, run
r = run("List all employee names", db, "q_happy", "employees")
```

**What exists:** `agent/graph.py` exposes only `graph`, `build_graph`, `AgentState`, `MAX_ITERATIONS`, and node functions. No `run()`.

**Note:** The reference repo (`original_repo/agent/graph.py`) also has no `run()` function. The test was written expecting a convenience wrapper that was never added to either codebase.

**Decision needed:** Add `run()` to `agent/graph.py`, OR rewrite tests to call `graph.invoke()` directly (matching the reference repo pattern).

---

## Gap 2 — `run()` signature uses a raw SQLite path, not a `db_id`

**What the test passes:**
```python
db = tempfile.mktemp(suffix=".db")   # e.g. /tmp/tmpXXXX.db
run("question", db, "q_happy", "employees")
```

**What `execute_sql` and `render_schema` expect:** a `db_id` string (e.g. `"financial"`) mapped to `data/bird/{db_id}.sqlite`.

**Impact:** Even if `run()` is added, calling `execute_sql(db, sql)` with a raw path like `/tmp/tmpXXXX.db` will try to open `data/bird//tmp/tmpXXXX.db.sqlite` — which will crash.

**Decision needed:** Either:
- Make `run()` detect raw paths and pass them directly to sqlite3 (bypassing `db_path()`), OR
- Rewrite tests to use a `db_id` pointing to a real BIRD database in `data/bird/`.

---

## Gap 3 — `run()` return dict format is not defined anywhere

**What the test reads from the return value:**
| Key | Used in | Assertion |
|-----|---------|-----------|
| `predicted_sql` | `test_happy_path` (print only) | None |
| `result` | `test_happy_path` (print only) | None |
| `final_status` | `test_happy_path` | `== "success"` |
| `iterations` | all tests (print + assert) | `>= 1` (revise), `<= MAX_ITERATIONS` (cap) |
| `latency_s` | all tests (print only) | None |
| `error` | `test_revise_path`, `test_iteration_cap` (print only) | None |

**Decision needed:** `run()` must assemble this dict from the graph's final state. Mapping:
- `predicted_sql` ← `state.sql`
- `result` ← `state.execution.rows` (or rendered string)
- `final_status` ← `"success"` if `state.execution.ok` else `"error"` (or `"failed"`)
- `iterations` ← `state.iteration` (raw counter from graph; note "want 0" comment in test_happy_path is aspirational, not asserted)
- `latency_s` ← wall-clock time of `graph.invoke()`
- `error` ← `state.execution.error`

---

## Gap 4 — `run()` 4th argument (`"employees"`) purpose unclear

The test passes a table name as the 4th argument but there is no such parameter in the reference agent interface. Possible interpretations:
1. Ignored entirely by `run()`.
2. Used to render a minimal schema from the raw SQLite file when `render_schema(db_id)` can't be used (since `db` is a temp path, not a `db_id`).

**Decision needed:** Clarify intent. If Gap 2 is resolved by detecting raw paths in `render_schema`, the 4th arg may be unused.

---

## Gap 5 — Tests make real LLM calls (no mocking)

All three tests call the LLM endpoint configured in `.env` (`https://api.tokenfactory.nebius.com/v1`). There is no mock/stub path. Tests will:
- Pass if Token Factory is reachable and the API key is valid.
- Fail with a connection/auth error otherwise.

This is intentional (integration tests), but worth noting for CI.

---

## Summary

| # | Gap | Blocker? | Fix in: |
|---|-----|----------|---------|
| 1 | `run()` missing | YES | `agent/graph.py` (add) or `tests/test_agent.py` (rewrite) |
| 2 | Raw path vs db_id mismatch | YES (once #1 is fixed) | `agent/graph.py` `run()` impl or `tests/test_agent.py` |
| 3 | Return dict format undefined | YES (once #1 is fixed) | `agent/graph.py` `run()` impl |
| 4 | 4th argument semantics | Minor | Clarify in `run()` signature |
| 5 | No LLM mocking | No (by design) | — |
