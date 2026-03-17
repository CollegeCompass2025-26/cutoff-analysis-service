import pdfplumber
import os

def inspect_mhtcet():
    path = "c:/cutoff-analysis-service/data/raw/mhtcet/2024_R1.pdf"
    if not os.path.exists(path):
        print("MHT-CET sample not found.")
        return
    print(f"--- Inspecting MHT-CET: {path} ---")
    with pdfplumber.open(path) as pdf:
        # Check first data page (usually after index)
        for i in range(min(5, len(pdf.pages))):
            print(f"Page {i+1}:")
            text = pdf.pages[i].extract_text()
            if text:
                print(text[:500])
            table = pdf.pages[i].extract_table()
            if table:
                print("Table header sample:", table[0])
                if len(table) > 1:
                    print("First row sample:", table[1])
            print("-" * 20)

def inspect_kcet():
    path = "c:/cutoff-analysis-service/data/raw/kcet/2024_R1.pdf"
    if not os.path.exists(path):
        print("KCET sample not found.")
        return
    print(f"--- Inspecting KCET: {path} ---")
    with pdfplumber.open(path) as pdf:
        for i in range(min(5, len(pdf.pages))):
            print(f"Page {i+1}:")
            text = pdf.pages[i].extract_text()
            if text:
                print(text[:500])
            table = pdf.pages[i].extract_table()
            if table:
                print("Table header sample:", table[0])
            print("-" * 20)

if __name__ == "__main__":
    inspect_mhtcet()
    inspect_kcet()
