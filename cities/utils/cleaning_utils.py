import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys


def find_repo_root():
    """
    Finds the repo root (fodler containing .gitignore) and adds it to sys.path.
    """
    current_dir = os.getcwd()
    while True:
        marker_file_path = os.path.join(current_dir, '.gitignore') 
        if os.path.isfile(marker_file_path):
            return current_dir 
            
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    return current_dir




def standardize_and_scale(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes and scales float columns in a DataFrame to [-1,1], copying other columns. Returns a new DataFrame.
    """
    standard_scaler = StandardScaler()

    new_data = pd.DataFrame()
    for column in data.columns:
        if data.dtypes[column] != 'float64':
            new_data[column] = data[column].copy()
        else:
            new = data[column].copy().values.reshape(-1, 1)
            new = standard_scaler.fit_transform(new)

        
            positive_mask = new >= 0
            negative_mask = new < 0

            min_positive = np.min(new[positive_mask])
            max_positive = np.max(new[positive_mask])
            scaled_positive = (new[positive_mask] - min_positive) / (max_positive - min_positive)

            min_negative = np.min(new[negative_mask])
            max_negative = np.max(new[negative_mask])
            scaled_negative = (new[negative_mask] - min_negative) / (max_negative - min_negative) - 1

            scaled_values = np.empty_like(new, dtype=float)
            scaled_values[positive_mask] = scaled_positive
            scaled_values[negative_mask] = scaled_negative


            new_data[column] =  scaled_values.reshape(-1)

    
    return new_data


def clean_gdp():
    gdp = pd.read_csv("data/raw/CAGDP1_2001_2021.csv", encoding='ISO-8859-1')

    gdp = gdp.loc[:9533] #drop notes at the bottom

    gdp['GeoFIPS'] = gdp['GeoFIPS'].fillna('').astype(str)
    gdp['GeoFIPS'] = gdp['GeoFIPS'].str.strip(' "').astype(int)


    #remove large regions
    gdp = gdp[gdp['GeoFIPS'] % 100 != 0]

    # focus on chain-type GDP
    mask = gdp['Description'].str.startswith('Chain')
    gdp = gdp[mask]

    #drop Region number, Tablename, LineCode, IndustryClassification columns (the last one is empty anyway)
    gdp = gdp.drop(gdp.columns[2:8], axis=1) 

    #2012 makes no sense, it's 100 throughout
    gdp = gdp.drop('2012', axis=1)

    gdp.replace('(NA)', np.nan, inplace=True)
    gdp.replace('(NM)', np.nan, inplace=True)


    #nan_rows = gdp[gdp.isna().any(axis=1)] #  if inspection is needed
    
    gdp.dropna(axis=0, inplace=True)

    for column in gdp.columns[2:]:
        gdp[column] = gdp[column].astype(float)


    assert gdp['GeoName'].is_unique

    for column in gdp.columns[2:]:
        assert (gdp[column] > 0).all(), f"Negative values in {column}"
        assert (gdp[column].isna().sum() == 0), f"Missing values in {column}"
        assert (gdp[column].isnull().sum() == 0), f"Null values in {column}"
        assert (gdp[column] < 3000).all(), f"Values suspiciously large in {column}"

    #TODO_Nikodem investigate strange large values

    gdp_wide = gdp.copy()
    gdp_long = pd.melt(gdp.copy(),  id_vars=['GeoFIPS', 'GeoName'],
    var_name='Year',
    value_name='Value')


    gdp_std_wide = standardize_and_scale(gdp)
    gdp_std_long = pd.melt(gdp_std_wide.copy(),  id_vars=['GeoFIPS', 'GeoName'],
                    var_name='Year', 
                    value_name='Value')

    gdp_wide.to_csv("data/processed/gdp_wide.csv", index=False)
    gdp_long.to_csv("data/processed/gdp_long.csv", index=False)
    gdp_std_wide.to_csv("data/processed/gdp_std_wide.csv", index=False)
    gdp_std_long.to_csv("data/processed/gdp_std_long.csv", index=False)


