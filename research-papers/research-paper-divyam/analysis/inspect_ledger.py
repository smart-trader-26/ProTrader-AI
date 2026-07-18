"""Inspect the prediction ledger schema and contents for the recalibration-law study."""
import sqlite3

DB = r"c:\Users\divya\Desktop\finance\data\ledger\predictions.sqlite"

con = sqlite3.connect(DB)
cur = con.cursor()
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("TABLES:", tables)
for t in tables:
    cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
    n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"\n== {t} (n={n}) ==")
    for c in cols:
        print("  ", c[1], c[2])
    rows = cur.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()
    for r in rows:
        print("  sample:", r)
