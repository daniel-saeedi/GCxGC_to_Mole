#!/bin/bash

# Define the path to the Python script
SCRIPT_PATH="codes/PubChem_Info_Extractor.py"

# Define the path to the input file containing the compounds with links
PATH_TO_LINKS="results/pubchem_links.csv"

# Define the path where the extracted information will be saved
PATH_TO_SAVE="results/pubchem_extracted_info.csv"

# Run the Python script with the provided arguments
python $SCRIPT_PATH --path_to_link $PATH_TO_LINKS --path_to_save $PATH_TO_SAVE
