from cities.utils.data_grabber import DataGrabber
from cities.utils.cleaning_utils import standardize_and_scale
import numpy as np
import pandas as pd



def clean_population():
    data = DataGrabber()
    data.get_gdp_wide()
    gdp = data.gdp_wide
    
    # grabbing gdp for comparison
    
    transport = pd.read_csv("data/raw/smartLocationSmall.csv")
    
    #choosing transport values
    transport = transport[['GeoFIPS', 'D3A']]

    assert transport.isna().sum().sum() == 0
    assert transport['GeoFIPS'].is_unique
    
    # subsetting to common FIPS numbers
    
    common_fips = np.intersect1d(gdp['GeoFIPS'].unique(),
                                 transport['GeoFIPS'].unique())
    transport = transport[transport['GeoFIPS'].isin(common_fips)]
    
    assert len(common_fips) == len(transport['GeoFIPS'].unique())
    assert len(transport) > 3000, 'The number of records is lower than 3000'

    # adding geoname column
    transport = transport.merge(gdp[['GeoFIPS', 'GeoName']],
                            on='GeoFIPS', how='left')[['GeoFIPS', 'GeoName', 'D3A']]
    
    patState = r', [A-Z]{2}(\*{1,2})?$'
    GeoNameError = 'Wrong Geoname value!'
    assert transport['GeoName'].str.contains(patState, regex=True).all(), GeoNameError
    assert sum(transport['GeoName'].str.count(', ')) == transport.shape[0], GeoNameError
    
    
    # changing values to floats
        
    for column in transport.columns[2:]:
        transport[column] = transport[column].astype(float)
        
    
    # Standardizing, formatting, saving
    
    transport_std_wide = standardize_and_scale(transport)
        
    transport_std_wide.to_csv("data/processed/transport_std_wide.csv", index=False)