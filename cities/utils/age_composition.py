import pandas as pd

from cities.utils.clean_variable import VariableCleaner
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()

data = DataGrabber()
data.get_features_wide(["gdp"])
gdp = data.wide["gdp"]


def clean_age_first():
    age = pd.read_csv(f"{root}/data/raw/age_composition.csv")

    age.iloc[:, 2:] = age.iloc[:, 2:].div(age["total_pop"], axis=0) * 100
    age.drop("total_pop", axis=1, inplace=True)

    age.to_csv(f"{root}/data/raw/age_percentages.csv", index=False)


def clean_age_composition():
    clean_age_first()

    cleaner = VariableCleaner(
        variable_name="age_composition",
        path_to_raw_csv=f"{root}/data/raw/age_percentages.csv",
        year_or_category="Category",
    )
    cleaner.clean_variable()
