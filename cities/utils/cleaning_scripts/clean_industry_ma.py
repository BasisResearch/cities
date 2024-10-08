from cities.utils.clean_variable import VariableCleanerMSA
from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


def clean_industry_ma():
    cleaner = VariableCleanerMSA(
        variable_name="industry_ma",
        path_to_raw_csv=f"{root}/data/raw/industry_ma.csv",
        year_or_category="Category",
    )
    cleaner.clean_variable()
