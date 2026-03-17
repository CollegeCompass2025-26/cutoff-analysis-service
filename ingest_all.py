import sys
import os
sys.path.append('c:/cutoff-analysis-service')
from app.core.extraction.db_loader import DBLoader

def run_ingestion():
    loader = DBLoader()
    
    # 1. Load MHT-CET
    mhtcet_csv = "c:/cutoff-analysis-service/data/processed/mhtcet/mhtcet_2024_r1_processed.csv"
    if os.path.exists(mhtcet_csv):
        print("Ingesting MHT-CET data...")
        loader.load_mhtcet_csv(mhtcet_csv)
    
    # 2. Load KCET (if exists)
    kcet_csv = "c:/cutoff-analysis-service/data/processed/kcet/kcet_2024_r1_processed.csv"
    if os.path.exists(kcet_csv):
        print("Ingesting KCET data...")
        loader.load_kcet_csv(kcet_csv)

    # 3. Verify counts
    conn = loader.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT exam_name, COUNT(*) FROM raw_cutoffs GROUP BY exam_name")
    counts = cur.fetchall()
    print("\n--- Current Database Stats ---")
    for exam, count in counts:
        print(f"{exam}: {count} records")
    cur.close()
    conn.close()

if __name__ == "__main__":
    run_ingestion()
