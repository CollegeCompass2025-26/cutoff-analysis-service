import pdfplumber
import os

def inspect_kcet():
    # Attempting various possible names/paths
    paths = [
        "c:/cutoff-analysis-service/data/raw/kcet/2024_R1.pdf",
        "c:/cutoff-analysis-service/data/raw/kcet/2023_R1.pdf"
    ]
    for path in paths:
        if os.path.exists(path):
            print(f"--- Inspecting KCET: {path} ---")
            with pdfplumber.open(path) as pdf:
                for i in range(min(5, len(pdf.pages))):
                    print(f"Page {i+1}:")
                    text = pdf.pages[i].extract_text()
                    if text: print(text[:300])
                    table = pdf.pages[i].extract_table()
                    if table: print("Table sample:", table[:3])
            return # Only need one
    print("No KCET files found to inspect.")

if __name__ == "__main__":
    inspect_kcet()
