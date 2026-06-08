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
