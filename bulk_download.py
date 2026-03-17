import requests
import os

mhtcet_links = {
    "2024_R1": "https://fe2025.mahacet.org/2024/2024ENGG_CAP1_CutOff.pdf",
    "2024_AI_R1": "https://fe2025.mahacet.org/2024/2024ENGG_CAP1_AI_CutOff.pdf",
    "2024_R2": "https://fe2025.mahacet.org/2024/2024ENGG_CAP2_CutOff.pdf",
    "2024_AI_R2": "https://fe2025.mahacet.org/2024/2024ENGG_CAP2_AI_CutOff.pdf",
    "2024_R3": "https://fe2025.mahacet.org/2024/2024ENGG_CAP3_CutOff.pdf",
    "2024_AI_R3": "https://fe2025.mahacet.org/2024/2024ENGG_CAP3_AI_CutOff.pdf",
    "2023_R1": "https://fe2024.mahacet.org/2023/2023ENGG_CAP1_CutOff.pdf",
    "2023_AI_R1": "https://fe2024.mahacet.org/2023/2023ENGG_CAP1_AI_CutOff.pdf",
    "2023_R2": "https://fe2024.mahacet.org/2023/2023ENGG_CAP2_CutOff.pdf",
    "2023_AI_R2": "https://fe2024.mahacet.org/2023/2023ENGG_CAP2_AI_CutOff.pdf",
    "2023_R3": "https://fe2024.mahacet.org/2023/2023ENGG_CAP3_CutOff.pdf",
    "2023_AI_R3": "https://fe2024.mahacet.org/2023/2023ENGG_CAP3_AI_CutOff.pdf",
    "2022_R1": "https://fe2023.mahacet.org/2022/2022ENGG_CAP1_CutOff.pdf",
    "2022_AI_R1": "https://fe2023.mahacet.org/2022/2022ENGG_CAP1_AI_CutOff.pdf",
    "2022_R2": "https://fe2023.mahacet.org/2022/2022ENGG_CAP2_CutOff.pdf",
    "2022_AI_R2": "https://fe2023.mahacet.org/2022/2022ENGG_CAP2_AI_CutOff.pdf",
    "2021_R1": "https://fe2022.mahacet.org/2021/2021ENGG_CAP1_CutOff.pdf",
    "2021_AI_R1": "https://fe2022.mahacet.org/2021/2021ENGG_CAP1_AI_CutOff.pdf",
    "2021_R2": "https://fe2022.mahacet.org/2021/2021ENGG_CAP2_CutOff.pdf",
    "2021_AI_R2": "https://fe2022.mahacet.org/2021/2021ENGG_CAP2_AI_CutOff.pdf",
}

kcet_links = {
    "2024_R1": "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2024/ENGG_CUTOFF_GEN_R1.pdf",
    "2024_R2": "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2024/ENGG_CUTOFF_2024_GEN_R2_FIN.pdf",
    "2023_R1": "https://cetonline.karnataka.gov.in/keawebentry456/cet2023/ENGG_CUTOFF_2023_GENkannada.pdf",
    "2023_R2Ext": "https://cetonline.karnataka.gov.in/keawebentry456/cet2023/ENR2_CUTGENkannada.pdf",
    "2022_R1": "https://cetonline.karnataka.gov.in/keawebentry456/cet2022/ENGG_CUT_GEN_2022.pdf",
    "2022_R2": "https://cetonline.karnataka.gov.in/keawebentry456/cet2022/ENGG_CUT_R2_2022.pdf",
    "2021_R1": "https://kea.kar.nic.in/cet2021/R1/engg_cutoff_gen.pdf",
    "2021_R2": "https://kea.kar.nic.in/cet2021/R2/engg_cutoff_gen.pdf",
}

def download_files(links, base_path):
    os.makedirs(base_path, exist_ok=True)
    for name, url in links.items():
        filename = f"{name}.pdf"
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            print(f"Skipping {filename}, already exists.")
            continue
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    download_files(mhtcet_links, "c:/cutoff-analysis-service/data/raw/mhtcet")
    download_files(kcet_links, "c:/cutoff-analysis-service/data/raw/kcet")
