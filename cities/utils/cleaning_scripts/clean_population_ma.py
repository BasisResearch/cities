from cities.utils.clean_variable import VariableCleanerMSA
from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


def clean_population_ma():
    cleaner = VariableCleanerMSA(
        variable_name="population_ma",
        path_to_raw_csv=f"{root}/data/raw/population_ma.csv",
        year_or_category="Year",
    )
    cleaner.clean_variable()
