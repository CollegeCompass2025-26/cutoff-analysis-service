import sys
import os
import pandas as pd
import pdfplumber
sys.path.append('c:/cutoff-analysis-service')
from app.core.extraction.pdf_extractor import PDFExtractor

class NEETExtractor(PDFExtractor):
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        super().__init__(raw_data_dir)
        self.processed_data_dir = processed_data_dir
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

    def extract_full_neet_report(self, filepath: str, output_name: str) -> str:
        """Extracts all pages from a NEET UG allotment PDF and saves to CSV."""
        all_data = []
        headers = ['S.No', 'Rank', 'Quota', 'Institute', 'Course', 'Category', 'Candidate_Category', 'Remarks']
        
        print(f"Opening {filepath}...")
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
            
            for i, page in enumerate(pdf.pages):
                # Skip the first ~30 pages as they are abbreviations/rules
                if i < 30: continue
                
                if i % 100 == 0:
                    print(f"Processing page {i}...")
                
                table = page.extract_table()
                if not table:
                    continue
                
                for row in table:
                    # Filter out header rows and empty rows
                    if not row or len(row) < 5: continue
                    if 'Rank' in str(row[1]) or 'S.No' in str(row[0]): continue
                    
                    # Ensure the row has the correct number of expected columns
                    # If the row is split or has extras, we might need more complex logic
                    # But for now, we'll try to keep the first 8 columns
                    all_data.append(row[:8])
        
        df = pd.DataFrame(all_data, columns=headers)
        # Final cleaning: Drop rows where Rank is not a number
        df = df[pd.to_numeric(df['Rank'].astype(str).str.replace(',', ''), errors='coerce').notnull()]
        
        output_path = os.path.join(self.processed_data_dir, output_name)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(all_data)} rows to {output_path}")
        return output_path

if __name__ == "__main__":
    neet_extractor = NEETExtractor(
        raw_data_dir="c:/cutoff-analysis-service/data/raw",
        processed_data_dir="c:/cutoff-analysis-service/data/processed"
    )
    
    # Process NEET 2024 Round 1 (already downloaded)
    raw_pdf = "c:/cutoff-analysis-service/data/raw/neet_ug_2024_r1.pdf"
    if os.path.exists(raw_pdf):
        neet_extractor.extract_full_neet_report(raw_pdf, "neet_ug_2024_r1_processed.csv")
    else:
        # Download and process
        neet_url_2024_r1 = "https://cdnbbsr.s3waas.gov.in/s3e0f7a4d0ef9b84b83b693bbf3feb8e6e/uploads/2024/08/2024082536.pdf"
        filepath = neet_extractor.download_pdf(neet_url_2024_r1, "neet_ug_2024_r1.pdf")
        if filepath:
            neet_extractor.extract_full_neet_report(filepath, "neet_ug_2024_r1_processed.csv")
