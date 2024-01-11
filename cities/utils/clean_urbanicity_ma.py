import numpy as np
import pandas as pd

from cities.utils.clean_variable import VariableCleanerMSA
from cities.utils.cleaning_utils import find_repo_root

root = find_repo_root()


def clean_urbanicity_initially():
    population_urban = pd.read_csv(
        f"{root}/data/raw/DECENNIALDHC2020.P2-2023-12-25T165149.csv"
    )

    population_urban.set_index("Label (Grouping)", inplace=True)
    transposed_df = population_urban.transpose()
    transposed_df.reset_index(inplace=True)
    df_population_urban = transposed_df.copy()

    filtered_df = pd.DataFrame(
        df_population_urban[df_population_urban["index"].str.endswith("Metro Area")]
    )

    filtered_df = filtered_df.rename(columns={"index": "MetroName"})

    filtered_df.columns = filtered_df.columns.str.replace("Total:", "total_pop")
    filtered_df.columns = filtered_df.columns.str.replace("Urban", "urban_pop")
    filtered_df.columns = filtered_df.columns.str.replace("Rural", "rural_pop")
    filtered_df = filtered_df.iloc[:, :-1].reset_index(drop=True)

    population_urban = filtered_df.copy()

    housing_urban = pd.read_csv(
        f"{root}/data/raw/DECENNIALDHC2020.H2-2023-12-25T174403.csv"
    )

    housing_urban.set_index("Label (Grouping)", inplace=True)
    transposed_df = housing_urban.transpose()
    transposed_df.reset_index(inplace=True)
    housing_urban = transposed_df.copy()

    filtered_df = pd.DataFrame(
        housing_urban[housing_urban["index"].str.endswith("Metro Area")]
    )

    filtered_df = filtered_df.rename(columns={"index": "MetroName"})

    filtered_df.columns = filtered_df.columns.str.replace("Total:", "total_housing")
    filtered_df.columns = filtered_df.columns.str.replace("Urban", "urban_housing")
    filtered_df.columns = filtered_df.columns.str.replace("Rural", "rural_housing")
    filtered_df = filtered_df.iloc[:, :-1].reset_index(drop=True)
    housing_urban = filtered_df.copy()

    metrolist = pd.read_csv(f"{root}/data/raw/metrolist.csv")

    merged_df = housing_urban.merge(population_urban, on="MetroName")

    merged_df["MetroName"] = merged_df["MetroName"].str.replace("Metro Area", "(MA)")

    df1_subset = metrolist[["GeoFIPS", "GeoName"]].drop_duplicates()

    merged_df = pd.merge(
        merged_df, df1_subset, left_on=["MetroName"], right_on=["GeoName"], how="left"
    )

    merged_df = merged_df.drop(columns=["GeoName"])
    merged_df.dropna(inplace=True)

    merged_df.columns = merged_df.columns.str.strip()
    ordered_columns = [
        "GeoFIPS",
        "MetroName",
        "total_housing",
        "urban_housing",
        "rural_housing",
        "total_pop",
        "urban_pop",
        "rural_pop",
    ]
    ordered_df = merged_df[ordered_columns]

    ordered_df = ordered_df.rename(columns={"MetroName": "GeoName"})

    numeric_columns = [
        "total_housing",
        "urban_housing",
        "rural_housing",
        "total_pop",
        "urban_pop",
        "rural_pop",
    ]
    ordered_df[numeric_columns] = (
        ordered_df[numeric_columns].replace({",": ""}, regex=True).astype(float)
    )

    ordered_df["GeoFIPS"] = ordered_df["GeoFIPS"].astype(np.int64)

    ordered_df["rural_pop_prct"] = ordered_df["rural_pop"] / ordered_df["total_pop"]
    ordered_df["rural_housing_prct"] = (
        ordered_df["rural_housing"] / ordered_df["total_housing"]
    )

    ordered_df.drop(["total_pop", "total_housing"], axis=1, inplace=True)

    ordered_df.reset_index(drop=True, inplace=True)

    ordered_df.to_csv(f"{root}/data/raw/urbanicity_ma.csv", index=False)


def clean_urbanicity_ma():
    clean_urbanicity_initially()

    cleaner = VariableCleanerMSA(
        variable_name="urbanicity_ma",
        path_to_raw_csv=f"{root}/data/raw/urbanicity_ma.csv",
        year_or_category="Category",
    )
    cleaner.clean_variable()
