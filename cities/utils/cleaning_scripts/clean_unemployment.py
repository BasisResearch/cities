from cities.utils.clean_variable import VariableCleaner
from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


def clean_unemployment():
    cleaner = VariableCleaner(
        variable_name="unemployment_rate",
        path_to_raw_csv=f"{root}/data/raw/unemployment_rate_wide_withNA.csv",
    )
    cleaner.clean_variable()
