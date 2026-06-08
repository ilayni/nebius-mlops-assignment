"""
Manual integration tests for A4 + A5.
Requires .env with OPENAI_BASE_URL, NEBIUS_API_KEY (or OPENAI_API_KEY),
VLLM_MODEL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.

Run from project root:
    python tests/test_agent.py
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agent.graph import MAX_ITERATIONS, run


def make_db(ddl: str, rows: list[tuple]) -> str:
    db = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db)
    conn.execute(ddl)
    conn.executemany(f"INSERT INTO {ddl.split('(')[0].split()[-1]} VALUES ({','.join(['?']*len(rows[0]))})", rows)
    conn.commit()
    conn.close()
    return db


def test_happy_path():
    print("=" * 55)
    print("TEST 1 — happy path (no revise expected)")
    print("=" * 55)
    db = make_db(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER)",
        [(1, "Alice", 90000), (2, "Bob", 75000)],
    )
    try:
        r = run("List all employee names", db, "q_happy", "employees")
        print(f"  SQL:        {r['predicted_sql']}")
        print(f"  Result:     {r['result']}")
        print(f"  Status:     {r['final_status']}")
        print(f"  Iterations: {r['iterations']}  (want 0)")
        print(f"  Latency:    {r['latency_s']}s")
        assert r["final_status"] == "success", f"Expected success, got {r['final_status']}"
        print("  PASS")
    finally:
        os.unlink(db)


def test_revise_path():
    print()
    print("=" * 55)
    print("TEST 2 — revise path (JOIN to non-existent table)")
    print("=" * 55)
    db = make_db(
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, amount REAL, status TEXT)",
        [(1, 99.9, "shipped"), (2, 45.0, "pending")],
    )
    try:
        # Asking for customer info that requires joining a table that doesn't exist.
        # Even the smartest model will generate a JOIN to a non-existent table,
        # which sqlite will reject with "no such table".
        r = run(
            "Show all orders along with the customer name and email from the customers table",
            db, "q_revise", "orders",
        )
        print(f"  SQL:        {r['predicted_sql']}")
        print(f"  Status:     {r['final_status']}")
        print(f"  Iterations: {r['iterations']}  (want >= 1)")
        print(f"  Error:      {r['error']}")
        print(f"  Latency:    {r['latency_s']}s")
        assert r["iterations"] >= 1, "Expected at least one revise iteration"
        print("  PASS")
    finally:
        os.unlink(db)


def test_iteration_cap():
    print()
    print("=" * 55)
    print("TEST 4 — iteration cap (pathological query, no crash)")
    print("=" * 55)
    db = make_db("CREATE TABLE t (x INTEGER)", [(1,), (2,)])
    try:
        r = run(
            "JOIN nonexistent_table ON fake_col = other_fake_col",
            db, "q_cap", "t",
        )
        print(f"  Status:     {r['final_status']}")
        print(f"  Iterations: {r['iterations']}  (want <= {MAX_ITERATIONS})")
        print(f"  Error:      {r['error']}")
        assert r["iterations"] <= MAX_ITERATIONS, "Exceeded MAX_ITERATIONS"
        print("  PASS")
    finally:
        os.unlink(db)


if __name__ == "__main__":
    print("Running A4/A5 integration tests...")
    print("(Test 3 — Langfuse trace — is manual: open localhost:3001 after these run)\n")
    test_happy_path()
    test_revise_path()
    test_iteration_cap()
    print()
    print("All tests done. Now check localhost:3001 for traces (TEST 3).")
