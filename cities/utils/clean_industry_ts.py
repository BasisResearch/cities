import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber


def clean_industry_ts():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide["gdp"]

    industry_ts = pd.read_csv("../data/raw/industry_time_series_people.csv")


    industry_ts["GEO_ID"] = industry_ts["GEO_ID"].str.split("US").str[1]
    industry_ts["GEO_ID"] = industry_ts["GEO_ID"].astype("int64")
    industry_ts = industry_ts.rename(columns={"GEO_ID": "GeoFIPS"})

    common_fips = np.intersect1d(gdp["GeoFIPS"].unique(), industry_ts["GeoFIPS"].unique())


    industry_ts = industry_ts[industry_ts["GeoFIPS"].isin(common_fips)]


    years = industry_ts['Year'].unique()

    for year in years:
        year_df = industry_ts[industry_ts['Year'] == year]
        missing_fips = set(common_fips) - set(year_df['GeoFIPS'])
        
        if missing_fips:
            missing_data = {'Year': [year] * len(missing_fips), 'GeoFIPS': list(missing_fips)}
            
            # Fill all columns from the fourth column (index 3) onward with 0
            for col in industry_ts.columns[2:]:
                missing_data[col] = 0
            
            missing_df = pd.DataFrame(missing_data)
            industry_ts = pd.concat([industry_ts, missing_df], ignore_index=True)

    industry_ts = industry_ts.merge(gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left")


    industry_ts = industry_ts[
        [
            "GeoFIPS",
            "GeoName",
            "Year",
            'agriculture_total',
            'mining_total',
            'construction_total',
            'manufacturing_total',
            'wholesale_trade_total',
            'retail_trade_total',
            'transportation_warehousing_total',
            'utilities_total',
            'information_total',
            'finance_insurance_total',
            'real_estate_total',
            'professional_services_total',
            'management_enterprises_total',
            'admin_support_services_total',
            'educational_services_total',
            'healthcare_social_services_total',
            'arts_recreation_total',
            'accommodation_food_services_total',
            'other_services_total',
            'public_administration_total',
        ]
    ]

    industry_ts = industry_ts.sort_values(by=["GeoFIPS", "GeoName", "Year"])



    columns_to_save = industry_ts.columns[industry_ts.columns.get_loc('Year') + 1:]


    for column in columns_to_save:

        selected_columns = ['GeoFIPS', 'GeoName', 'Year', column]
        subsetindustry_ts = industry_ts[selected_columns]

        subsetindustry_ts.rename(columns={column: 'Value'}, inplace=True)
        

        subsetindustry_ts_long = subsetindustry_ts.copy()

        file_name_long = f"industry_{column}_long.csv"
        subsetindustry_ts_long.to_csv(f"../data/processed/{file_name_long}", index=False)



        subsetindustry_ts_std_long = standardize_and_scale(subsetindustry_ts)

        file_name_std = f"industry_{column}_std_long.csv"
        subsetindustry_ts_std_long.to_csv(f"../data/processed/{file_name_std}", index=False)


        subsetindustry_ts_wide = subsetindustry_ts.pivot_table(
        index=["GeoFIPS", "GeoName"], columns="Year", values="Value"
        )
        subsetindustry_ts_wide.reset_index(inplace=True)
        subsetindustry_ts_wide.columns.name = None

        
        file_name_wide = f"industry_{column}_wide.csv"
        subsetindustry_ts_wide.to_csv(f"../data/processed/{file_name_wide}", index=False)
        

        subsetindustry_ts_std_wide = subsetindustry_ts_std_long.pivot_table(
        index=["GeoFIPS", "GeoName"], columns="Year", values="Value"
        )
        subsetindustry_ts_std_wide.reset_index(inplace=True)
        subsetindustry_ts_std_wide.columns.name = None

        file_name_std_wide = f"industry_{column}_std_wide.csv"
        subsetindustry_ts_std_wide.to_csv(f"../data/processed/{file_name_std_wide}", index=False)