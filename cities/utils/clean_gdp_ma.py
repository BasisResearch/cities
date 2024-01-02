from cities.utils.clean_variable import VariableCleanerMSA
from cities.utils.cleaning_utils import find_repo_root

root = find_repo_root()


def clean_gdp_ma():
    cleaner = VariableCleanerMSA(
        variable_name="gdp", path_to_raw_csv=f"{root}/data/raw/gdp_ma.csv"
    )
    cleaner.clean_variable()
