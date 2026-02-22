import sqlite3
import pandas as pd

DB_PATH = "dq_repository/dq_repository.db"

def load_readability_results() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)

    query = """
    SELECT *
    FROM dqresults
    WHERE dq_dimension = 'Readability'
    """

    df = pd.read_sql_query(query, con)
    con.close()

    return df
