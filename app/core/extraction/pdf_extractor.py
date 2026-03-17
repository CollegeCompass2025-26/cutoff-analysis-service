import pdfplumber
import requests
import os
import pandas as pd
from typing import List, Optional

class PDFExtractor:
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir
        if not os.path.exists(self.raw_data_dir):
            os.makedirs(self.raw_data_dir)

    def download_pdf(self, url: str, filename: str) -> Optional[str]:
        """Downloads a PDF from a URL and saves it locally."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            filepath = os.path.join(self.raw_data_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return filepath
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None

    def extract_tables(self, filepath: str) -> List[pd.DataFrame]:
        """Extracts tables from a PDF using pdfplumber."""
        all_tables = []
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        all_tables.append(df)
            return all_tables
        except Exception as e:
            print(f"Error extracting tables from {filepath}: {e}")
            return []

    def process_neet_ug_result(self, filepath: str) -> pd.DataFrame:
        """Specific extraction logic for NEET UG Allotment Results."""
        # Placeholder for complex table parsing
        # NEET results usually have: S.No, Roll No, Rank, Allotted Institute, Course, Allotted Category, Candidate Category, Remarks
        tables = self.extract_tables(filepath)
        if not tables:
            return pd.DataFrame()
        return pd.concat(tables, ignore_index=True)

# Example usage
if __name__ == "__main__":
    extractor = PDFExtractor(raw_data_dir="c:/cutoff-analysis-service/data/raw")
    # Test with a NEET 2024 PDF if needed
