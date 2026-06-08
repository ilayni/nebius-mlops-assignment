from functools import lru_cache
from pathlib import Path
import sqlite3

ROOT = Path(__file__).parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    p = Path(db_id)
    if p.is_absolute() and p.exists():
        return p
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
