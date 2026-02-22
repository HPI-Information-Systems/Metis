from pathlib import Path
import sqlite3
import os

p = Path("dq_repository/dq_repository.db")
print("CWD:", os.getcwd())
print("DB exists:", p.exists())
print("DB absolute:", p.resolve())
print("DB size:", p.stat().st_size if p.exists() else None)

con = sqlite3.connect(str(p))
cur = con.cursor()
tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("TABLES:", tables)
con.close()
