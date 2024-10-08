from cities.utils.clean_variable import VariableCleaner
from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


def clean_homeownership():
    variables = [
        "median_owner_occupied_home_value",
        "median_rent",
        "homeownership_rate",
    ]

    for variable in variables:
        cleaner = VariableCleaner(
            variable_name=variable,
            path_to_raw_csv=f"{root}/data/raw/{variable}.csv",
            year_or_category="Category",
        )
        cleaner.clean_variable()
