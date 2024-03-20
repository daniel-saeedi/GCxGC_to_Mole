
import argparse
import numpy as np
import pandas as pd
import requests
import time

def get_pubchem_link(compound_name):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    search_url = f"{base_url}/compound/name/{compound_name}/cids/JSON"

    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()
        if "IdentifierList" in data and "CID" in data["IdentifierList"]:
            cid = data["IdentifierList"]["CID"][0]
            link = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
            return link
        else:
            return "NaN"
    else:
        return "NaN"

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract PubChem link to compounds')
    parser.add_argument('--path_to_compounds', type=str, help='Path to all compounds numpy file')
    parser.add_argument('--path_to_save', type=str, help='Path to save the extracted information')

    args = parser.parse_args()
    compounds = np.load(args.path_to_compounds)

    info_extracted = []
    for compound in compounds:
        link = get_pubchem_link(compound)
        print(f"{compound}: {link}")
        info_extracted.append((compound, link))
        info_extracted_df = pd.DataFrame(info_extracted, columns=["Compound", "PubChem_Link"])
        info_extracted_df.to_csv(args.path_to_save)
        # time.sleep(5)