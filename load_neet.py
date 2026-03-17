import sys
import os
sys.path.append('c:/cutoff-analysis-service')
from app.core.extraction.db_loader import DBLoader

def load_data():
    loader = DBLoader()
    
    # 1. Load JoSAA (Already done in previous command, but for completeness)
    # 2. Load NEET 2024
    neet_csv = "c:/cutoff-analysis-service/data/processed/neet_ug_2024_r1_processed.csv"
    if os.path.exists(neet_csv):
        print(f"Loading NEET data from {neet_csv}...")
        loader.load_neet_csv(neet_csv, 2024, "Round 1")
    else:
        print(f"NEET CSV not found: {neet_csv}")

if __name__ == "__main__":
    load_data()
