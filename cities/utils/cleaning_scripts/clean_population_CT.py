from cities.utils.clean_variable import VariableCleanerCT
from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


def clean_population_CT():
    cleaner = VariableCleanerCT(
        variable_name="population_pre2020_CT",
        path_to_raw_csv=f"{root}/data/raw/pop_pre2020_filtered_wide.csv",
        year_or_category="Year",
        time_interval="pre2020",
    )
    cleaner.clean_variable()

    cleaner2 = VariableCleanerCT(
        variable_name="population_post2020_CT",
        path_to_raw_csv=f"{root}/data/raw/pop_post2020_filtered.csv",
        year_or_category="Year",
        time_interval="post2020",
    )
    cleaner2.clean_variable()
