import os
import psycopg2
from dotenv import load_dotenv
import json

load_dotenv("c:/cutoff-analysis-service/.env")

def dump_schema():
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    schema_info = {}
    
    # 1. Get all tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
    """)
    tables = [r[0] for r in cur.fetchall()]
    schema_info['tables'] = tables
    
    # 2. Get columns and row counts for each table
    schema_info['details'] = {}
    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """)
            columns = [{"name": r[0], "type": r[1]} for r in cur.fetchall()]
            
            schema_info['details'][table] = {
                "row_count": count,
                "columns": columns
            }
        except Exception as e:
            schema_info['details'][table] = {"error": str(e)}
            conn.rollback()
            
    # 3. Get foreign keys (simplified)
    cur.execute("""
        SELECT
            tc.table_name, kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
        WHERE constraint_type = 'FOREIGN KEY';
    """)
    fks = cur.fetchall()
    schema_info['foreign_keys'] = [{"table": r[0], "column": r[1], "foreign_table": r[2], "foreign_column": r[3]} for r in fks]

    with open('c:/cutoff-analysis-service/db_schema_dump.json', 'w') as f:
        json.dump(schema_info, f, indent=2)
        
    cur.close()
    conn.close()
    print("Schema dumped to c:/cutoff-analysis-service/db_schema_dump.json")

if __name__ == "__main__":
    dump_schema()
