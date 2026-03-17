import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def init_db():
    db_url = os.getenv("DATABASE_URL")
    sql_file = "c:/cutoff-analysis-service/db_schema.sql"
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    with open(sql_file, 'r') as f:
        sql = f.read()
        cur.execute("DROP TABLE IF EXISTS raw_cutoffs CASCADE;")
        cur.execute("DROP TABLE IF EXISTS colleges_metadata CASCADE;")
        cur.execute(sql)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
