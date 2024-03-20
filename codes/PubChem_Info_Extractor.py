
import argparse
import numpy as np
import pandas as pd
import requests
import time

def get_chemical_info(pubchem_url):
    # Extract the CID (Compound ID) from the URL
    cid = pubchem_url.split('/')[-1]

    # Construct the API URL to fetch the compound information
    api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChI,InChIKey,CanonicalSMILES/JSON"

    # Send a GET request to the API
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        properties = data['PropertyTable']['Properties'][0]

        # Extract InChI, InChIKey, and Canonical SMILES
        inchi = properties.get('InChI', 'Not available')
        inchikey = properties.get('InChIKey', 'Not available')
        canonical_smiles = properties.get('CanonicalSMILES', 'Not available')

        return inchi, inchikey, canonical_smiles
    else:
        return None, None, None

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract PubChem info from links')
    parser.add_argument('--path_to_link', type=str, help='Path to all compounds with links csv file')
    parser.add_argument('--path_to_save', type=str, help='Path to save the extracted information')

    args = parser.parse_args()
    compounds = pd.read_csv(args.path_to_link)

    info_extracted = []
    for idx, compound in compounds.iterrows():
        compound_name = compound['Compound']
        link = compound['PubChem_Link']
        print(f'Link: {link}')

        if pd.isna(link):
            inchi, inchikey, canonical_smiles = ('NaN', 'NaN', 'NaN')
        else:
            inchi, inchikey, canonical_smiles = get_chemical_info(link)

        print(f"{compound_name}: ",inchi, inchikey, canonical_smiles)
        info_extracted.append((compound_name, link,inchi, inchikey, canonical_smiles))
        info_extracted_df = pd.DataFrame(info_extracted, columns=["Compound", "PubChem_Link", "InChI", "InChIKey", "Canonical_SMILES"])
        info_extracted_df.to_csv(args.path_to_save, index=False)
        time.sleep(3)