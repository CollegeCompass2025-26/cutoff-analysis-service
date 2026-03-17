import sys
import os
sys.path.append('c:/cutoff-analysis-service')
from app.core.extraction.pdf_extractor import PDFExtractor

def test_extraction():
    extractor = PDFExtractor(raw_data_dir="c:/cutoff-analysis-service/data/raw")
    neet_url_2024_r1 = "https://cdnbbsr.s3waas.gov.in/s3e0f7a4d0ef9b84b83b693bbf3feb8e6e/uploads/2024/08/2024082536.pdf"
    filename = "neet_ug_2024_r1.pdf"
    
    print(f"Downloading {filename}...")
    filepath = extractor.download_pdf(neet_url_2024_r1, filename)
    
    if filepath:
        print(f"Downloaded to {filepath}. Extracting sample tables...")
        # Only extract first 5 pages to avoid massive output
        with pdfplumber.open(filepath) as pdf:
            for i in range(min(5, len(pdf.pages))):
                page = pdf.pages[i]
                table = page.extract_table()
                if table:
                    print(f"Page {i+1} sample data:")
                    print(table[0]) # Header
                    print(table[1]) # First row
    else:
        print("Download failed.")

if __name__ == "__main__":
    import pdfplumber
    test_extraction()
