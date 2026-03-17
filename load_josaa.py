import sys
import os
sys.path.append('c:/cutoff-analysis-service')
from app.core.extraction.db_loader import DBLoader

def load_josaa_data():
    loader = DBLoader()
    base_dir = "c:/cutoff-analysis-service/data/josaa"
    files = [
        ("2021.csv", 2021),
        ("2022.csv", 2022),
        ("2023.csv", 2023),
        ("2024_Round1.csv", 2024)
    ]
    
    for filename, year in files:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            print(f"Starting load for {filename}...")
            loader.load_josaa_csv(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    load_josaa_data()
