import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()

def clean_population():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide["gdp"]

    cainc30 = pd.read_csv(f"{root}/data/raw/CAINC30_1969_2021.csv", encoding="ISO-8859-1")

    population = cainc30[cainc30["Description"] == " Population (persons) 3/"].copy()

    population["GeoFIPS"] = population["GeoFIPS"].fillna("").astype(str)
    population["GeoFIPS"] = population["GeoFIPS"].str.strip(' "').astype(int)

    population = population[population["GeoFIPS"] % 1000 != 0]

    common_fips = np.intersect1d(
        population["GeoFIPS"].unique(), gdp["GeoFIPS"].unique()
    )
    assert len(common_fips) == len(gdp["GeoFIPS"].unique())

    population = population[population["GeoFIPS"].isin(common_fips)]
    assert population.shape[0] == gdp.shape[0]

    order = gdp["GeoFIPS"].tolist()
    population = population.set_index("GeoFIPS").reindex(order).reset_index()

    # align with gdp
    assert population["GeoFIPS"].tolist() == gdp["GeoFIPS"].tolist()
    assert population["GeoName"].is_unique

    population = population.drop(population.columns[2:8], axis=1)
    assert population.shape[0] == gdp.shape[0]

    # 243 NAs prior to 1993
    # na_counts = (population == '(NA)').sum().sum()
    # print(na_counts)

    population.replace("(NA)", np.nan, inplace=True)
    population.replace("(NM)", np.nan, inplace=True)

    # removed years prior to 1993, missigness, long time ago
    population = population.drop(population.columns[2:26], axis=1)

    assert population.isna().sum().sum() == 0
    assert population.shape[0] == gdp.shape[0]

    for column in population.columns[2:]:
        population[column] = population[column].astype(float)

    assert population.shape[0] == gdp.shape[0]

    population_long = pd.melt(
        population.copy(),
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Year",
        value_name="Value",
    )

    population_std_wide = standardize_and_scale(population)
    population_std_long = pd.melt(
        population_std_wide.copy(),
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Year",
        value_name="Value",
    )

    population.to_csv(f"{root}/data/processed/population_wide.csv", index=False)
    population_long.to_csv(f"{root}/data/processed/population_long.csv", index=False)
    population_std_wide.to_csv(f"{root}/data/processed/population_std_wide.csv", index=False)
    population_std_long.to_csv(f"{root}/data/processed/population_std_long.csv", index=False)
