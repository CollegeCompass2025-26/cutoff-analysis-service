import pdfplumber
import sys
import os

filepath = "c:/cutoff-analysis-service/data/raw/neet_ug_2024_r1.pdf"
with pdfplumber.open(filepath) as pdf:
    # Check page 50 for real data
    page = pdf.pages[50]
    table = page.extract_table()
    if table:
        print("Page 51 Table Sample:")
        for row in table[:10]:
            print(row)
    else:
        print("No table found on page 51")
