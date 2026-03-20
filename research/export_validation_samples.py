import os
import sys
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Set path for imports
sys.path.append(os.getcwd())
load_dotenv("c:/cutoff-analysis-service/.env")

def save_validation_samples():
    print("📡 Fetching 1000 balanced validation samples from DB...")
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        # We sample 1000 random records to represent the evaluation baseline
        query = "SELECT * FROM ml_features_v3 WHERE closing_rank > 0 ORDER BY RANDOM() LIMIT 1000"
        df = pd.read_sql(query, conn)
        conn.close()
        
        output_path = 'research/validation_samples_1000.csv'
        df.to_csv(output_path, index=False)
        print(f"✅ Validation samples saved to {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    save_validation_samples()
