import numpy as np
from cities.utils.data_grabber import find_repo_root, MSADataGrabberCSV, DataGrabberCSV
import pandas as pd
import numpy as np
import requests
from us import states

root = find_repo_root()

print('Warning: The process will take around 15min.')

# S2301_C04_001E Estimate!!Unemployment rate!!Population 16 years and over
variables =  "NAME,S2301_C04_001E"
county_fips = "*" # all counties
tract = "*" # all tracts
api_key = '077d857d6c12d5b9b3aeafa07d2c1916ba12a86c' # private api key required to access the data https://api.census.gov/data/key_signup.html
# year = 2022

interval = list(range(2010, 2023))
dfs = []

for year in interval:
    for x in range(0, len(states.STATES)): # in this call it's not possible to use the '*' wildcard to access all states, so we need to iterate over all states
        fips = states.STATES[x].fips

        url = f'https://api.census.gov/data/{year}/acs/acs5/subject?get={variables}&for=tract:{tract}&in=state:{fips}&in=county:{county_fips}&key={api_key}'

        response = requests.get(url)

        assert response.status_code == 200, 'The data retrieval went wrong'  # 200 means success

        print(f'{fips} fips for year {year} done')

        data = response.json()

        df = pd.DataFrame(data[1:], columns=data[0])
        df['Year'] = year  # Add the year column

        dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)


unemployment_combined = combined_df.copy()

column_mapping = {
    'S2301_C04_001E': 'Value'
}

unemployment_combined.rename(columns=column_mapping, inplace=True)

state_abbreviations = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

unemployment_combined['GeoFIPS'] = unemployment_combined.apply(lambda row: f"{row['state']}{row['county']}{row['tract']}", axis=1).astype(np.int64)

unemployment_combined.drop(['state', 'county', 'tract'], axis=1, inplace=True)

def parse_geo_name(name):
    if ';' in name:
        parts = name.split(';')
    else:
        parts = name.split(',')

    if len(parts) >= 3:
        county = parts[1].strip().replace(' County', '')
        state_full = parts[2].strip()
        state_abbr = state_abbreviations.get(state_full, state_full)  
        return f"{county}, {state_abbr} (CT)"
    return "Unknown"


unemployment_combined['GeoName'] = unemployment_combined['NAME'].apply(parse_geo_name).astype(str)

assert unemployment_combined[unemployment_combined['GeoName'] == 'Unknown'].shape[0] == 0, 'There are Unknown GeoNames'

unemployment_combined = unemployment_combined.drop(['NAME'], axis=1)

unemployment_combined.sort_values(by=['Year', 'GeoFIPS', 'GeoName'], inplace=True)
unemployment_combined = unemployment_combined[['GeoFIPS', 'GeoName', 'Year', 'Value']].reset_index(drop=True)

unemployment_combined_pre2020 = unemployment_combined[unemployment_combined['Year'] < 2020].reset_index(drop=True)
unemployment_combined_post2020 = unemployment_combined[unemployment_combined['Year'] >= 2020].reset_index(drop=True)


geo_counts = unemployment_combined_pre2020['GeoFIPS'].value_counts()
geo_in_all_years = geo_counts[geo_counts == geo_counts.max()].index.tolist()
unemployment_combined_pre2020_filtered = unemployment_combined_pre2020[unemployment_combined_pre2020['GeoFIPS'].isin(geo_in_all_years)]
missin_count = unemployment_combined_pre2020['GeoFIPS'].nunique() - unemployment_combined_pre2020_filtered['GeoFIPS'].nunique()

print(f" {missin_count} GeoFIPS values were removed from the pre-2020 data")


geo_counts = unemployment_combined_post2020['GeoFIPS'].value_counts()
geo_in_all_years = geo_counts[geo_counts == geo_counts.max()].index.tolist()
unemployment_combined_post2020_filtered = unemployment_combined_post2020[unemployment_combined_post2020['GeoFIPS'].isin(geo_in_all_years)]
missin_count = unemployment_combined_post2020['GeoFIPS'].nunique() - unemployment_combined_post2020_filtered['GeoFIPS'].nunique()

print(f" {missin_count} GeoFIPS values were removed from the post-2020 data")

unemployment_combined_post2020_filtered_wide = unemployment_combined_post2020_filtered.pivot(index=['GeoFIPS', 'GeoName'], columns='Year', values='Value')
unemployment_combined_post2020_filtered_wide = unemployment_combined_post2020_filtered_wide.reset_index()
unemployment_combined_post2020_filtered_wide.columns.name = None

unemployment_combined_pre2020_filtered_wide = unemployment_combined_pre2020_filtered.pivot(index=['GeoFIPS', 'GeoName'], columns='Year', values='Value')
unemployment_combined_pre2020_filtered_wide = unemployment_combined_pre2020_filtered_wide.reset_index()
unemployment_combined_pre2020_filtered_wide.columns.name = None

unemployment_combined_pre2020_filtered_wide = unemployment_combined_pre2020_filtered_wide.dropna(how='any')
unemployment_combined_post2020_filtered_wide = unemployment_combined_post2020_filtered_wide.dropna(how='any')

columns_to_convert = unemployment_combined_pre2020_filtered_wide.columns[2:]  
unemployment_combined_pre2020_filtered_wide[columns_to_convert] = unemployment_combined_pre2020_filtered_wide[columns_to_convert].astype(float)

columns_to_convert = unemployment_combined_post2020_filtered_wide.columns[2:]  
unemployment_combined_post2020_filtered_wide[columns_to_convert] = unemployment_combined_post2020_filtered_wide[columns_to_convert].astype(float)

print(f"Pre-2020 data shape: {unemployment_combined_pre2020_filtered_wide.shape}")
print(f"Post-2020 data shape: {unemployment_combined_post2020_filtered_wide.shape}")

unemployment_combined_pre2020_filtered_wide.to_csv(f"{root}/data/raw/unemployment_pre2020_ct.csv", index=False)
unemployment_combined_post2020_filtered_wide.to_csv(f"{root}/data/raw/unemployment_post2020_ct.csv", index=False)
