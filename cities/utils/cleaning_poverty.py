import pandas as pd
import os
from pathlib import Path

from cities.utils.clean_variable import clean_variable



path = Path(__file__).parent.absolute()


poverty_variables = [ 'povertyAll', 'povertyAllprct', 
        'povertyUnder18', 'povertyUnder18prct', 'medianHouseholdIncome']

def clean_poverty():
    for variable_name in poverty_variables:

        path_to_raw_csv = os.path.join(
            path, f"../../data/raw/{variable_name}_wide.csv")
    
    
        clean_variable(variable_name, path_to_raw_csv)