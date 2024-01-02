import numpy as np
import pandas as pd

from cities.utils.clean_variable import VariableCleanerMSA
from cities.utils.cleaning_utils import find_repo_root

root = find_repo_root()


def clean_ethnic_initially():
    ethnic_composition = pd.read_csv(f"{root}/data/raw/ethnic_composition_cbsa.csv")
    metro_areas = pd.read_csv(f"{root}/data/raw/metrolist.csv")

    ethnic_composition["CBSA"] = ethnic_composition["CBSA"].astype(np.int64)
    ethnic_composition = ethnic_composition[
        ethnic_composition["CBSA"].isin(metro_areas["GeoFIPS"])
    ]

    ethnic_composition = pd.merge(
        ethnic_composition,
        metro_areas[["GeoFIPS", "GeoName"]],
        left_on="CBSA",
        right_on="GeoFIPS",
        how="inner",
    )
    ethnic_composition = ethnic_composition.drop_duplicates(subset=["CBSA"])

    ethnic_composition.drop(columns="CBSA", inplace=True)

    cols_to_save = ethnic_composition.shape[1] - 2
    ethnic_composition_ma = ethnic_composition[
        ["GeoFIPS", "GeoName"] + list(ethnic_composition.columns[0:cols_to_save])
    ]

    ethnic_composition_ma.iloc[:, 2:] = ethnic_composition_ma.iloc[:, 2:].apply(
        pd.to_numeric, errors="coerce"
    )
    ethnic_composition_ma[ethnic_composition_ma.columns[2:]] = ethnic_composition_ma[
        ethnic_composition_ma.columns[2:]
    ].astype(float)

    ethnic_composition_ma["other_race_races"] = (
        ethnic_composition_ma["other_race"] + ethnic_composition_ma["two_or_more_sum"]
    )
    ethnic_composition_ma = ethnic_composition_ma.drop(
        ["other_race", "two_or_more_sum"], axis=1
    )

    ethnic_composition_ma["totalALT"] = ethnic_composition_ma.iloc[:, 3:].sum(axis=1)
    assert (
        ethnic_composition_ma["totalALT"] == ethnic_composition_ma["total_pop"]
    ).all()
    ethnic_composition_ma = ethnic_composition_ma.drop("totalALT", axis=1)

    row_sums = ethnic_composition_ma.iloc[:, 2:].sum(axis=1)
    ethnic_composition_ma.iloc[:, 3:] = ethnic_composition_ma.iloc[:, 3:].div(
        row_sums, axis=0
    )

    ethnic_composition_ma = ethnic_composition_ma.drop(["total_pop"], axis=1)

    ethnic_composition_ma.to_csv(
        f"{root}/data/raw/ethnic_composition_ma.csv", index=False
    )


def clean_ethnic_composition_ma():
    clean_ethnic_initially()

    cleaner = VariableCleanerMSA(
        variable_name="ethnic_composition",
        path_to_raw_csv=f"{root}/data/raw/ethnic_composition_ma.csv",
    )
    cleaner.clean_variable()
