import numpy as np
from cities.utils.data_grabber import find_repo_root, MSADataGrabberCSV, DataGrabberCSV
import pandas as pd
import numpy as np
import requests
from us import states

root = find_repo_root()

variables = "NAME,S1901_C01_013E,S1901_C01_012E"
county_fips = "*"  # all counties
tract = "*"  # all tracts
api_key = '077d857d6c12d5b9b3aeafa07d2c1916ba12a86c'  # Your private API key
years = [2019, 2022]

dfs = []

for year in years:
    for x in range(len(states.STATES)):  # Iterate over all states
        fips = states.STATES[x].fips

        url = f'https://api.census.gov/data/{year}/acs/acs5/subject?get={variables}&for=tract:{tract}&in=state:{fips}&in=county:{county_fips}&key={api_key}'
        
        response = requests.get(url)

        assert response.status_code == 200, 'The data retrieval went wrong'  # 200 means success

        print(f'{fips} fips done for year {year}')
        
        data = response.json()

        df = pd.DataFrame(data[1:], columns=data[0])
        df['Year'] = year  # Add the year column

        dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

income = combined_df.copy()

columns_income = {
    'S1901_C01_012E': "median_income",
    'S1901_C01_013E': "mean_income",
}

income.rename(columns=columns_income, inplace=True)

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

income['GeoFIPS'] = income.apply(lambda row: f"{row['state']}{row['county']}{row['tract']}", axis=1).astype(np.int64)

income.drop(['state', 'county', 'tract'], axis=1, inplace=True)


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


income['GeoName'] = income['NAME'].apply(parse_geo_name).astype(str)

assert income[income['GeoName'] == 'Unknown'].shape[0] == 0, 'There are Unknown GeoNames'

income = income.drop(['NAME'], axis=1)

income.sort_values(by=['Year', 'GeoFIPS', 'GeoName'], inplace=True)
income = income[['GeoFIPS', 'GeoName', 'Year', 'mean_income', 'median_income']].reset_index(drop=True)

income_pre2020 = income[income['Year'] < 2020].reset_index(drop=True).drop(['Year'], axis=1)
income_post2020 = income[income['Year'] >= 2020].reset_index(drop=True).drop(['Year'], axis=1)

income_pre2020 = income_pre2020.dropna(how='any')
income_post2020 = income_post2020.dropna(how='any')

columns_to_convert = income_pre2020.columns[2:]  
income_pre2020[columns_to_convert] = income_pre2020[columns_to_convert].astype(float)

columns_to_convert = income_post2020.columns[2:]  
income_post2020[columns_to_convert] = income_post2020[columns_to_convert].astype(float)

print(f"Pre-2020 data shape: {income_pre2020.shape}")

print(f"Post-2020 data shape: {income_post2020.shape}")

income_pre2020.to_csv(f"{root}/data/raw/income_pre2020_ct.csv", index=False)
income_post2020.to_csv(f"{root}/data/raw/income_post2020_ct.csv", index=False)