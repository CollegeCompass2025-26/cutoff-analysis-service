import sys
import os
import pandas as pd
import pdfplumber
import re
sys.path.append('c:/cutoff-analysis-service')
from app.core.extraction.pdf_extractor import PDFExtractor

class MHTCETExtractor(PDFExtractor):
    def __init__(self, raw_data_dir: str = "data/raw/mhtcet", processed_data_dir: str = "data/processed/mhtcet"):
        super().__init__(raw_data_dir)
        self.processed_data_dir = processed_data_dir
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

    def extract_mhtcet_report(self, filepath: str, output_name: str) -> str:
        all_data = []
        current_college = "Unknown"
        current_course = "Unknown"
        
        print(f"Opening MHT-CET Report: {filepath}...")
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                if i % 50 == 0: print(f"Processing page {i}...")
                
                text = page.extract_text()
                if not text: continue
                
                # Try to find College Name (Pattern: 4 digits code - Name)
                college_match = re.search(r'(\d{4})\s-\s(.*?)\s+(?:Government|Private|University)', text)
                if college_match:
                    current_college = college_match.group(2).strip()
                
                # Try to find Course Name
                course_match = re.search(r'\d{9}\s-\s(.*?)(?:\n|$)', text)
                if course_match:
                    current_course = course_match.group(1).strip()

                table = page.extract_table()
                if not table: continue
                
                for row in table:
                    # Look for data rows. Typically have 'I' or 'II' etc in first column
                    if not row or len(row) < 3: continue
                    if row[0] not in ['I', 'II', 'III', 'IV']: continue
                    
                    # Columns usually: Round, Rank(Percentile), SeatType, ...
                    for j in range(1, len(row)):
                        cell = row[j]
                        if not cell: continue
                        
                        # MHT-CET format: "Rank\n(Percentile)" e.g. "95351\n(63.5370419)"
                        if '\n(' in str(cell):
                            parts = str(cell).split('\n(')
                            rank = parts[0].strip()
                            percentile = parts[1].replace(')', '').strip()
                            seat_type = "UNKNOWN" # Need better header mapping
                            
                            all_data.append({
                                'exam_name': 'MHT-CET',
                                'college_name': current_college,
                                'course_name': current_course,
                                'round': row[0],
                                'rank': rank,
                                'percentile': percentile,
                                'category': seat_type
                            })
        
        df = pd.DataFrame(all_data)
        output_path = os.path.join(self.processed_data_dir, output_name)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(all_data)} rows to {output_path}")
        return output_path

if __name__ == "__main__":
    extractor = MHTCETExtractor()
    sample = "c:/cutoff-analysis-service/data/raw/mhtcet/2024_R1.pdf"
    if os.path.exists(sample):
        extractor.extract_mhtcet_report(sample, "mhtcet_2024_r1_processed.csv")
