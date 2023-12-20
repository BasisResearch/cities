
from cities.utils.clean_variable import VariableCleaner
from cities.utils.cleaning_utils import find_repo_root
root = find_repo_root()


def clean_population_ma():
    cleaner = VariableCleaner(variable_name="population",
                              path_to_raw_csv = f"{root}/data/raw/population_ma.csv",
                              YearOrCategory="Year", region_type="MA")
    cleaner.clean_variable()


