import numpy as np
import pandas as pd

from cities.utils.clean_gdp import clean_gdp
from cities.utils.cleaning_utils import find_repo_root, standardize_and_scale
from cities.utils.data_grabber import DataGrabber


class VariableCleaner:
    def __init__(
        self, variable_name: str, path_to_raw_csv: str, YearOrCategory: str = "Year"
    ):  # why are we using this parameter?
        self.variable_name = variable_name
        self.path_to_raw_csv = path_to_raw_csv
        self.YearOrCategory = YearOrCategory
        self.root = find_repo_root()
        self.data_grabber = DataGrabber()
        self.metro_areas = None
        self.gdp = None
        self.variable_db = None

    def clean_variable(self):
        self.load_raw_csv()
        self.drop_nans()
        self.load_gdp_data()
        self.check_exclusions()
        self.restrict_common_fips()
        self.save_csv_files(self.region_type)

    def load_raw_csv(self):
        self.variable_db = pd.read_csv(self.path_to_raw_csv)
        self.variable_db["GeoFIPS"] = self.variable_db["GeoFIPS"].astype(int)

    def drop_nans(self):
        self.variable_db = self.variable_db.dropna()

    def load_gdp_data(self):
        self.data_grabber.get_features_wide(["gdp"])
        self.gdp = self.data_grabber.wide["gdp"]

    def check_exclusions(self):
        common_fips = np.intersect1d(
            self.gdp["GeoFIPS"].unique(), self.variable_db["GeoFIPS"].unique()
        )
        if (
            len(
                np.setdiff1d(
                    self.gdp["GeoFIPS"].unique(), self.variable_db["GeoFIPS"].unique()
                )
            )
            > 0
        ):
            self.add_new_exclusions(common_fips)
            clean_gdp()
            self.clean_variable()

    def add_new_exclusions(self, common_fips):
        new_exclusions = np.setdiff1d(
            self.gdp["GeoFIPS"].unique(), self.variable_db["GeoFIPS"].unique()
        )
        print("Adding new exclusions to exclusions.csv: " + str(new_exclusions))
        exclusions = pd.read_csv((f"{self.root}/data/raw/exclusions.csv"))
        new_rows = pd.DataFrame(
            {
                "dataset": [self.variable_name] * len(new_exclusions),
                "exclusions": new_exclusions,
            }
        )
        exclusions = pd.concat([exclusions, new_rows], ignore_index=True)
        exclusions = exclusions.drop_duplicates()
        exclusions = exclusions.sort_values(by=["dataset", "exclusions"]).reset_index(
            drop=True
        )
        exclusions.to_csv((f"{self.root}/data/raw/exclusions.csv"), index=False)
        print("Rerunning gdp cleaning with new exclusions")

    def restrict_common_fips(self):
        common_fips = np.intersect1d(
            self.gdp["GeoFIPS"].unique(), self.variable_db["GeoFIPS"].unique()
        )
        self.variable_db = self.variable_db[
            self.variable_db["GeoFIPS"].isin(common_fips)
        ]
        self.variable_db = self.variable_db.merge(
            self.gdp[["GeoFIPS", "GeoName"]], on=["GeoFIPS", "GeoName"], how="left"
        )
        self.variable_db = self.variable_db.sort_values(by=["GeoFIPS", "GeoName"])
        for column in self.variable_db.columns:
            if column not in ["GeoFIPS", "GeoName"]:
                self.variable_db[column] = self.variable_db[column].astype(float)

    def save_csv_files(self, regions):
        # it would be great to make sure that a db is wide, if not make it wide
        variable_db_wide = self.variable_db.copy()
        variable_db_long = pd.melt(
            self.variable_db,
            id_vars=["GeoFIPS", "GeoName"],
            var_name=self.YearOrCategory,
            value_name="Value",
        )
        variable_db_std_wide = standardize_and_scale(self.variable_db)
        variable_db_std_long = pd.melt(
            variable_db_std_wide.copy(),
            id_vars=["GeoFIPS", "GeoName"],
            var_name=self.YearOrCategory,
            value_name="Value",
        )

        variable_db_wide.to_csv(
            (f"{self.root}/data/processed/" + self.variable_name + "_wide.csv"),
            index=False,
        )
        variable_db_long.to_csv(
            (f"{self.root}/data/processed/" + self.variable_name + "_long.csv"),
            index=False,
        )
        variable_db_std_wide.to_csv(
            (f"{self.root}/data/processed/" + self.variable_name + "_std_wide.csv"),
            index=False,
        )
        variable_db_std_long.to_csv(
            (f"{self.root}/data/processed/" + self.variable_name + "_std_long.csv"),
            index=False,
        )


class VariableCleanerMSA(
    VariableCleaner
):  # this class inherits functionalites of VariableCleaner, but its adjusted to MSA level
    def clean_variable(self):
        self.load_raw_csv()
        self.drop_nans()
        self.process_data()
        # self.check_exclusions('MA') functionality to implement in the future
        self.save_csv_files()

    def load_metro_areas(self):
        self.metro_areas = pd.read_csv(f"{self.root}/data/raw/metrolist.csv")

    def process_data(self):
        self.load_metro_areas()
        assert (
            self.metro_areas["GeoFIPS"].nunique()
            == self.variable_db["GeoFIPS"].nunique()
        )
        assert (
            self.metro_areas["GeoName"].nunique()
            == self.variable_db["GeoName"].nunique()
        )
        self.variable_db["GeoFIPS"] = self.variable_db["GeoFIPS"].astype(np.int64)

    def save_csv_files(self):
        # wrangling
        variable_db_wide = self.variable_db.copy()
        variable_db_long = pd.melt(
            self.variable_db,
            id_vars=["GeoFIPS", "GeoName"],
            var_name=self.YearOrCategory,
            value_name="Value",
        )
        variable_db_std_wide = standardize_and_scale(self.variable_db)
        variable_db_std_long = pd.melt(
            variable_db_std_wide.copy(),
            id_vars=["GeoFIPS", "GeoName"],
            var_name=self.YearOrCategory,
            value_name="Value",
        )
        # saving
        variable_db_wide.to_csv(
            (f"{self.root}/data/MSA_level/" + self.variable_name + "_ma_wide.csv"),
            index=False,
        )
        variable_db_long.to_csv(
            (f"{self.root}/data/MSA_level/" + self.variable_name + "_ma_long.csv"),
            index=False,
        )
        variable_db_std_wide.to_csv(
            (f"{self.root}/data/MSA_level/" + self.variable_name + "_ma_std_wide.csv"),
            index=False,
        )
        variable_db_std_long.to_csv(
            (f"{self.root}/data/MSA_level/" + self.variable_name + "_ma_std_long.csv"),
            index=False,
        )
