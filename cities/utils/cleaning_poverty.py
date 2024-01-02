from cities.utils.clean_variable import VariableCleaner
from cities.utils.cleaning_utils import find_repo_root

root = find_repo_root()


poverty_variables = [
    "povertyAll",
    "povertyAllprct",
    "povertyUnder18",
    "povertyUnder18prct",
    "medianHouseholdIncome",
]


def clean_poverty():
    for variable_name in poverty_variables:
        cleaner = VariableCleaner(
            variable_name,
            path_to_raw_csv=f"{root}data/raw/{variable_name}_wide.csv",
            year_or_category="Year",
            region_type="MA",
        )
        cleaner.clean_variable()
