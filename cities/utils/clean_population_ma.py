from cities.utils.clean_variable import VariableCleanerMSA
from cities.utils.cleaning_utils import find_repo_root

root = find_repo_root()


def clean_population_ma():
    cleaner = VariableCleanerMSA(
        variable_name="population",
        path_to_raw_csv=f"{root}/data/raw/population_ma.csv",
        YearOrCategory="Year",
    )
    cleaner.clean_variable()
