import os
import psycopg2
from dotenv import load_dotenv

load_dotenv("c:/cutoff-analysis-service/.env")

def inspect_db():
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # List tables
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = cur.fetchall()
    print(f"Tables: {tables}")
    
    for table_tuple in tables:
        table = table_tuple[0]
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"Table {table}: {count} rows")
        
        # Get columns
        cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'")
        columns = cur.fetchall()
        print(f"Columns in {table}: {columns}")
        
    # Check years and exams in raw_cutoffs
    cur.execute("SELECT exam_name, year, COUNT(*) FROM raw_cutoffs GROUP BY exam_name, year ORDER BY exam_name, year")
    distribution = cur.fetchall()
    print(f"Data Distribution (Exam, Year, Count): {distribution}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    inspect_db()
