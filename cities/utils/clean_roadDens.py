from cities.utils.data_grabber import DataGrabber
from cities.utils.cleaning_utils import standardize_and_scale
import numpy as np
import pandas as pd



def clean_population():
    data = DataGrabber()
    data.get_gdp_wide()
    gdp = data.gdp_wide
    
    smart = pd.read_csv("../data/raw/EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv")

    smart['STATEFP'] = smart['STATEFP'].astype(str).str.zfill(2)
    smart['COUNTYFP'] = smart['COUNTYFP'].astype(str).str.zfill(3)
    smart['GeoFIPS'] = smart['STATEFP'] + smart['COUNTYFP']

    smart['GeoFIPS'] = smart['GeoFIPS'].astype(int)
    smart = smart.sort_values('GeoFIPS')
    
    roadDensRegions = smart[['GeoFIPS', 'D3A', 'Ac_Land']]
    
    # converting into square miles
    roadDensRegions['MilesSq_Land'] = roadDensRegions['Ac_Land'] * 0.0015625 
    roadDensRegions['roads'] = roadDensRegions['D3A'] * roadDensRegions['MilesSq_Land']

    roadDens = roadDensRegions.groupby('GeoFIPS').agg({'roads': 'sum', 'MilesSq_Land': 'sum'})
    roadDens['Rdens'] = roadDens['roads'] / roadDens['MilesSq_Land']

    roadDens = roadDens.reset_index()
    roadDens = roadDens[['GeoFIPS', 'Rdens']]

    assert roadDens['GeoFIPS'].is_unique
    assert roadDens.isna().sum().sum() == 0
    
    # add Geoname? then join
    # to gdp ???
    
    # to float
    
    roadDens['column'] = roadDens['Rdens'].astype(float)
    
    

    
    
    # save in all formats (2 standardized)