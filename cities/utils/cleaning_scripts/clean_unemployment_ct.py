from cities.utils.clean_variable import VariableCleanerCT
from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


def clean_unemployment_CT():
    cleaner = VariableCleanerCT(
        variable_name="unemployment_pre2020_CT",
        path_to_raw_csv=f"{root}/data/raw/unemployment_pre2020_ct.csv",
        year_or_category_column_label="Year",
        time_interval="pre2020",
    )
    cleaner.clean_variable()

    cleaner2 = VariableCleanerCT(
        variable_name="unemployment_post2020_CT",
        path_to_raw_csv=f"{root}/data/raw/unemployment_post2020_ct.csv",
        year_or_category_column_label="Year",
        time_interval="post2020",
    )
    cleaner2.clean_variable()
