import argparse
import numpy as np
import pandas as pd
import requests
import time

def get_nist_link(compound_name):
    search_url = f"https://webbook.nist.gov/cgi/cbook.cgi?Name={compound_name}&Units=SI"
    response = requests.get(search_url)
    if response.status_code == 200:
        if "Name Not Found" not in response.text:
            return search_url
        else:
            return "Compound not found"
    else:
        return "Error accessing NIST Chemistry WebBook"

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract NIST link to compounds')
    parser.add_argument('--path_to_compounds', type=str, help='Path to all compounds numpy file')
    parser.add_argument('--path_to_save', type=str, help='Path to save the extracted information')

    args = parser.parse_args()
    compounds = np.load(args.path_to_compounds)

    info_extracted = []
    for compound in compounds:
        link = get_nist_link(compound)
        print(f"{compound}: {link}")
        info_extracted.append((compound, link))
        info_extracted_df = pd.DataFrame(info_extracted, columns=["Compound", "NIST_Link"])
        info_extracted_df.to_csv(args.path_to_save, index=False)
        
        time.sleep(1)