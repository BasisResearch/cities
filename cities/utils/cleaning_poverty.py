import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber




def clean_poverty():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide["gdp"]

    poverty = pd.read_csv("../data/raw/poverty.csv")




    poverty.fillna(0, inplace=True)

    columns_to_save = poverty.columns[poverty.columns.get_loc("Year") + 1 :]

    for column in columns_to_save:
        selected_columns = ["GeoFIPS", "GeoName", "Year", column]
        subsetpoverty = poverty[selected_columns]

        subsetpoverty.rename(columns={column: "Value"}, inplace=True)

        subsetpoverty_long = subsetpoverty.copy()

        file_name_long = f"industry_{column}_long.csv"
        subsetpoverty_long.to_csv(
            f"../data/processed/{file_name_long}", index=False
        )

        subsetpoverty_std_long = standardize_and_scale(subsetpoverty)
        subsetpoverty_std_long.fillna(0, inplace=True)

        file_name_std = f"industry_{column}_std_long.csv"
        subsetpoverty_std_long.to_csv(
            f"../data/processed/{file_name_std}", index=False
        )

        subsetpoverty_wide = subsetpoverty.pivot_table(
            index=["GeoFIPS", "GeoName"], columns="Year", values="Value"
        )
        subsetpoverty_wide.reset_index(inplace=True)
        subsetpoverty_wide.columns.name = None

        file_name_wide = f"industry_{column}_wide.csv"
        subsetpoverty_wide.to_csv(
            f"../data/processed/{file_name_wide}", index=False
        )

        subsetpoverty_std_wide = subsetpoverty_std_long.pivot_table(
            index=["GeoFIPS", "GeoName"], columns="Year", values="Value"
        )

        subsetpoverty_std_wide.fillna(0, inplace=True)

        subsetpoverty_std_wide.reset_index(inplace=True)
        subsetpoverty_std_wide.columns.name = None

        file_name_std_wide = f"industry_{column}_std_wide.csv"
        subsetpoverty_std_wide.to_csv(
            f"../data/processed/{file_name_std_wide}", index=False
        )