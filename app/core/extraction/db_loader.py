import pandas as pd
import psycopg2
from psycopg2 import extras
import os
from dotenv import load_dotenv

load_dotenv()

class DBLoader:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in .env")

    def get_connection(self):
        return psycopg2.connect(self.db_url)

    def load_neet_csv(self, filepath: str, year: int, round_name: str):
        """Loads processed NEET CSV into raw_cutoffs table."""
        df = pd.read_csv(filepath)
        print(f"Loading {len(df)} rows from {filepath}...")

        # NEET columns mapping: 
        # S.No, Roll No, Rank, Allotted Institute, Course, Allotted Category, Candidate Category, Remarks
        # Note: Headers might vary slightly by year, need cleaning
        
        conn = self.get_connection()
        cur = conn.cursor()

        # Mapping logic (adjust based on actual CSV columns)
        query = """
            INSERT INTO raw_cutoffs (
                exam_name, year, round, college_name, course_name, category, closing_rank
            ) VALUES %s
        """

        data = []
        for _, row in df.iterrows():
            try:
                # Basic mapping for NEET results
                college_name = row.get('Allotted Institute', row.get('Institute', 'Unknown'))
                course_name = row.get('Course', 'Unknown')
                category = row.get('Allotted Category', 'GEN')
                rank = str(row.get('Rank', row.get('All India Rank', '0'))).replace(',', '')
                rank = int(rank) if str(rank).isdigit() else 0

                data.append((
                    'NEET', year, round_name, college_name, course_name, category, rank
                ))
            except Exception as e:
                continue

        extras.execute_values(cur, query, data)
        conn.commit()
        cur.close()
        conn.close()
        print("Load complete.")

    def load_josaa_csv(self, filepath: str):
        """Loads processed JoSAA CSV into raw_cutoffs table."""
        df = pd.read_csv(filepath)
        print(f"Loading {len(df)} rows from {filepath}...")

        conn = self.get_connection()
        cur = conn.cursor()

        query = """
            INSERT INTO raw_cutoffs (
                exam_name, year, round, institute_type, college_name, course_name, quota, category, gender, opening_rank, closing_rank
            ) VALUES %s
        """

        data = []
        for _, row in df.iterrows():
            try:
                data.append((
                    'JoSAA', 
                    row['year'], 
                    str(row['round']), 
                    row['type'], 
                    row['institute'], 
                    row['program'], 
                    row['quota'], 
                    row['category'], 
                    row['gender'], 
                    int(row['orank']), 
                    int(row['crank'])
                ))
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        extras.execute_values(cur, query, data)
        conn.commit()
        cur.close()
        conn.close()
        print(f"Load complete for {filepath}.")

    def load_mhtcet_csv(self, filepath: str):
        """Loads processed MHT-CET CSV into raw_cutoffs table."""
        df = pd.read_csv(filepath)
        print(f"Loading {len(df)} MHT-CET rows from {filepath}...")
        
        conn = self.get_connection()
        cur = conn.cursor()
        
        query = """
            INSERT INTO raw_cutoffs (exam_name, year, round, college_name, course_name, percentile, category)
            VALUES %s
        """
        
        data = []
        for _, row in df.iterrows():
            try:
                data.append((
                    'MHT-CET',
                    2024, # Adjust as needed
                    str(row['round']),
                    row['college_name'],
                    row['course_name'],
                    float(row['percentile']),
                    row['category']
                ))
            except Exception as e:
                continue
                
        extras.execute_values(cur, query, data)
        conn.commit()
        cur.close()
        conn.close()
        print(f"Load complete for {filepath}.")

    def load_kcet_csv(self, filepath: str):
        """Loads processed KCET CSV into raw_cutoffs table."""
        df = pd.read_csv(filepath)
        print(f"Loading {len(df)} KCET rows from {filepath}...")
        
        conn = self.get_connection()
        cur = conn.cursor()
        
        query = """
            INSERT INTO raw_cutoffs (exam_name, year, round, college_name, course_name, closing_rank, category)
            VALUES %s
        """
        
        data = []
        for _, row in df.iterrows():
            try:
                data.append((
                    'KCET',
                    int(row['year']),
                    'Round 1',
                    row['college_name'],
                    row['course_name'],
                    int(row['closing_rank']),
                    row['category']
                ))
            except Exception as e:
                continue
                
        extras.execute_values(cur, query, data)
        conn.commit()
        cur.close()
        conn.close()
        print(f"Load complete for {filepath}.")
