import sqlite3

con = sqlite3.connect("dq_repository/dq_repository.db")
cur = con.cursor()

print("TABLES:")
print(cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall())

print("\nTOTAL ROWS:")
print(cur.execute("SELECT COUNT(1) FROM dqresults").fetchone())

print("\nREADABILITY GROUPED:")
print(cur.execute("""
    SELECT dq_metric, dq_granularity, COUNT(1)
    FROM dqresults
    WHERE dq_dimension='Readability'
    GROUP BY dq_metric, dq_granularity
    ORDER BY dq_metric
""").fetchall())

con.close()
