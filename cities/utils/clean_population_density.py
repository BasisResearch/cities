from cities.utils.clean_variable import VariableCleaner
from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


def clean_population_density():
    cleaner = VariableCleaner(
        variable_name="population_density",
        path_to_raw_csv=f"{root}/data/raw/population_density.csv",
    )
    cleaner.clean_variable()
