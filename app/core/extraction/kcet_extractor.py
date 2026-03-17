import sys
import os
import pandas as pd
import pdfplumber
import re
sys.path.append('c:/cutoff-analysis-service')
from app.core.extraction.pdf_extractor import PDFExtractor

class KCETExtractor(PDFExtractor):
    def __init__(self, raw_data_dir: str = "data/raw/kcet", processed_data_dir: str = "data/processed/kcet"):
        super().__init__(raw_data_dir)
        self.processed_data_dir = processed_data_dir
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

    def extract_kcet_report(self, filepath: str, output_name: str) -> str:
        all_data = []
        current_college = "Unknown"
        current_college_code = "Unknown"
        current_course = "Unknown"
        categories = []
        
        print(f"Opening KCET Report: {filepath}...")
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                if i % 20 == 0: print(f"Processing page {i}...")
                
                text = page.extract_text()
                if not text: continue
                
                # KCET pattern: E007 Dayananda Sagar...
                college_match = re.search(r'([A-Z]\d{3})\s+(.*?)(?:\n|$)', text)
                if college_match:
                    current_college_code = college_match.group(1).strip()
                    current_college = college_match.group(2).strip()
                    print(f"DEBUG: Found College: {current_college_code} - {current_college} on page {i}")
                
                table = page.extract_table()
                if not table: 
                    # print(f"DEBUG: No table on page {i}")
                    continue
                
                print(f"DEBUG: Found table with {len(table)} rows on page {i}")
                
                found_header_on_page = False
                for row_idx, row in enumerate(table):
                    # Category header check
                    if row and ('1G' in row or 'GM' in row or '2AG' in row):
                        categories = [c.strip() if c else None for c in row]
                        # print(f"Found Categories: {categories}")
                        found_header_on_page = True
                        data_rows = table[row_idx + 1:]
                        
                        for d_row in data_rows:
                            if not d_row or not d_row[0]: continue
                            
                            # If row starts with course name (not a rank)
                            if len(d_row[0]) > 6 and not d_row[0].strip().isdigit():
                                current_course = d_row[0].strip()
                                # print(f"Found Course: {current_course}")
                            
                            # Process columns
                            for col_idx in range(1, len(d_row)):
                                if col_idx < len(categories) and categories[col_idx]:
                                    val = str(d_row[col_idx]).strip()
                                    if val and val.isdigit():
                                        year_match = re.search(r'20\d{2}', filepath)
                                        year_val = year_match.group(0) if year_match else "2024"
                                        
                                        all_data.append({
                                            'exam_name': 'KCET',
                                            'year': year_val,
                                            'college_name': f"{current_college_code} - {current_college}",
                                            'course_name': current_course,
                                            'category': categories[col_idx],
                                            'closing_rank': int(val)
                                        })
                
                # If no header on this page but we have college/course, 
                # we might still have data rows if the table continued
                if not found_header_on_page and categories:
                    for d_row in table:
                        if not d_row or not d_row[0]: continue
                        if len(d_row[0]) > 6 and not d_row[0].strip().isdigit():
                            current_course = d_row[0].strip()
                        for col_idx in range(1, len(d_row)):
                            if col_idx < len(categories) and categories[col_idx]:
                                val = str(d_row[col_idx]).strip()
                                if val and val.isdigit():
                                    all_data.append({
                                        'exam_name': 'KCET',
                                        'year': '2024',
                                        'college_name': f"{current_college_code} - {current_college}",
                                        'course_name': current_course,
                                        'category': categories[col_idx],
                                        'closing_rank': int(val)
                                    })
        
        if not all_data:
            print("No data extracted from KCET report.")
            return ""
            
        df = pd.DataFrame(all_data)
        output_path = os.path.join(self.processed_data_dir, output_name)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(all_data)} rows to {output_path}")
        return output_path

if __name__ == "__main__":
    extractor = KCETExtractor()
    sample = "c:/cutoff-analysis-service/data/raw/kcet/2024_R2.pdf"
    if os.path.exists(sample):
        extractor.extract_kcet_report(sample, "kcet_2024_r1_processed.csv")
    else:
        print(f"Sample file not found at: {os.path.abspath(sample)}")
