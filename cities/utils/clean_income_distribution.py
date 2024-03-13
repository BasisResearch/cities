from cities.utils.clean_variable import VariableCleaner
from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


def clean_income_distribution():
    cleaner = VariableCleaner(
        variable_name="income_distribution",
        path_to_raw_csv=f"{root}/data/raw/income_distribution.csv",
        year_or_category="Category",
    )
    cleaner.clean_variable()
