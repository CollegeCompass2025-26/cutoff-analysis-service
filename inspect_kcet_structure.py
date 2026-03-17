import pdfplumber
import os

def inspect_kcet():
    # Use actual filenames found
    path = "c:/cutoff-analysis-service/data/raw/kcet/2024_R1.pdf"
    if not os.path.exists(path):
        print(f"KCET sample not found at {path}.")
        return
    print(f"--- Inspecting KCET: {path} ---")
    with pdfplumber.open(path) as pdf:
        for i in range(min(10, len(pdf.pages))):
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

if __name__ == "__main__":
    inspect_kcet()
