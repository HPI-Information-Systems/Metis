import sqlite3

con = sqlite3.connect("dq_repository/dq_repository.db")
cur = con.cursor()

rows = cur.execute("""
SELECT dq_metric, dq_granularity, COUNT(1)
FROM dqresults
WHERE dq_dimension='Readability'
GROUP BY dq_metric, dq_granularity
ORDER BY dq_metric, dq_granularity
""").fetchall()

print("ROWS:", len(rows))
for r in rows:
    print(r)

con.close()
