#!/bin/bash

# Define the path to the Python script
PYTHON_SCRIPT="codes/PubChem_Link_Extractor.py"

# Define the path to the compounds numpy file
PATH_TO_COMPOUNDS="results/all_compounds_array.npy"

# Define the path to save the extracted information
PATH_TO_SAVE="results/pubchem_links.csv"

# Run the Python script with the specified arguments
python $PYTHON_SCRIPT --path_to_compounds $PATH_TO_COMPOUNDS --path_to_save $PATH_TO_SAVE
