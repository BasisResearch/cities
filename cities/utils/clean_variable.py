import numpy as np
import pandas as pd

from cities.utils.clean_gdp import clean_gdp
from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber, find_repo_root


class VariableCleaner:
    def __init__(
        self, variable_name: str, path_to_raw_csv: str, year_or_category: str = "Year" # Year or Category
    ):
        self.variable_name = variable_name
        self.path_to_raw_csv = path_to_raw_csv
        self.year_or_category = year_or_category
        self.root = find_repo_root()
        self.data_grabber = DataGrabber()
        self.folder = 'processed'
        self.gdp = None
        self.variable_df = None

    def clean_variable(self):
        self.load_raw_csv()
        self.drop_nans()
        self.load_gdp_data()
        self.check_exclusions()
        self.restrict_common_fips()
        self.save_csv_files(self.folder)

    def load_raw_csv(self):
        self.variable_df = pd.read_csv(self.path_to_raw_csv)
        self.variable_df["GeoFIPS"] = self.variable_df["GeoFIPS"].astype(int)

    def drop_nans(self):
        self.variable_df = self.variable_df.dropna()

    def load_gdp_data(self):
        self.data_grabber.get_features_wide(["gdp"])
        self.gdp = self.data_grabber.wide["gdp"]

    def add_new_exclusions(self, common_fips):
        new_exclusions = np.setdiff1d(
            self.gdp["GeoFIPS"].unique(), self.variable_df["GeoFIPS"].unique()
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

    def check_exclusions(self):
        common_fips = np.intersect1d(
            self.gdp["GeoFIPS"].unique(), self.variable_df["GeoFIPS"].unique()
        )
        if (
            len(
                np.setdiff1d(
                    self.gdp["GeoFIPS"].unique(), self.variable_df["GeoFIPS"].unique()
                )
            )
            > 0
        ):
            self.add_new_exclusions(common_fips)
            clean_gdp()
            self.clean_variable()

    def restrict_common_fips(self):
        common_fips = np.intersect1d(
            self.gdp["GeoFIPS"].unique(), self.variable_df["GeoFIPS"].unique()
        )
        self.variable_df = self.variable_df[
            self.variable_df["GeoFIPS"].isin(common_fips)
        ]
        self.variable_df = self.variable_df.merge(
            self.gdp[["GeoFIPS", "GeoName"]], on=["GeoFIPS", "GeoName"], how="left"
        )
        self.variable_df = self.variable_df.sort_values(by=["GeoFIPS", "GeoName"])
        for column in self.variable_df.columns:
            if column not in ["GeoFIPS", "GeoName"]:
                self.variable_df[column] = self.variable_df[column].astype(float)

    def save_csv_files(self, folder):
        # it would be great to make sure that a db is wide, if not make it wide
        variable_db_wide = self.variable_df.copy()
        variable_db_long = pd.melt(
            self.variable_df,
            id_vars=["GeoFIPS", "GeoName"],
            var_name=self.year_or_category,
            value_name="Value",
        )
        variable_db_std_wide = standardize_and_scale(self.variable_df)
        variable_db_std_long = pd.melt(
            variable_db_std_wide.copy(),
            id_vars=["GeoFIPS", "GeoName"],
            var_name=self.year_or_category,
            value_name="Value",
        )

        variable_db_wide.to_csv(
            (f"{self.root}/data/{folder}/" + self.variable_name + "_wide.csv"),
            index=False,
        )
        variable_db_long.to_csv(
            (f"{self.root}/data/{folder}/" + self.variable_name + "_long.csv"),
            index=False,
        )
        variable_db_std_wide.to_csv(
            (f"{self.root}/data/{folder}/" + self.variable_name + "_std_wide.csv"),
            index=False,
        )
        variable_db_std_long.to_csv(
            (f"{self.root}/data/{folder}/" + self.variable_name + "_std_long.csv"),
            index=False,
        )


class VariableCleanerMSA(VariableCleaner):  # this class inherits functionalites of VariableCleaner, but works at the MSA level
    
    def __init__(self, variable_name: str, path_to_raw_csv: str, year_or_category: str = "Year"):
        super().__init__(variable_name, path_to_raw_csv, year_or_category)
        self.folder = 'MSA_level'
        self.metro_areas = None
    
    def clean_variable(self):
        self.load_raw_csv()
        self.drop_nans()
        self.process_data()
        # TODO self.check_exclusions('MA') functionality needs to be implemented in the future
        # TODO but only if data missigness turns out to be a serious problem
        # for now, process_data runs a check and reports missingness
        # but we need to be more careful about MSA missingnes handling
        # as there are much fewer MSAs than counties
        self.save_csv_files(self.folder)

    def load_metro_areas(self):
        self.metro_areas = pd.read_csv(f"{self.root}/data/raw/metrolist.csv")

    def process_data(self):
        self.load_metro_areas()
        assert (
            self.metro_areas["GeoFIPS"].nunique()
            == self.variable_df["GeoFIPS"].nunique()
        )
        assert (
            self.metro_areas["GeoName"].nunique()
            == self.variable_df["GeoName"].nunique()
        )
        self.variable_df["GeoFIPS"] = self.variable_df["GeoFIPS"].astype(np.int64)

   
