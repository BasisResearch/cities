import numpy as np
import pandas as pd

from cities.utils.cleaning_scripts.clean_gdp import clean_gdp
from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber, find_repo_root


class VariableCleaner:
    def __init__(
        self,
        variable_name: str,
        path_to_raw_csv: str,
        year_or_category_column_label: str = "Year",  # Column name to store years or categories in the long format
    ):
        self.variable_name = variable_name
        self.path_to_raw_csv = path_to_raw_csv
        self.year_or_category_column_label = year_or_category_column_label
        self.root = find_repo_root()
        self.data_grabber = DataGrabber()
        self.folder = "processed"
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
        self.variable_df = pd.read_csv(
            self.path_to_raw_csv
        )  # changed from int to np.int64
        self.variable_df["GeoFIPS"] = self.variable_df["GeoFIPS"].astype(np.int64)

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
            var_name=self.year_or_category_column_label,
            value_name="Value",
        )
        variable_db_std_wide = standardize_and_scale(self.variable_df)
        variable_db_std_long = pd.melt(
            variable_db_std_wide.copy(),
            id_vars=["GeoFIPS", "GeoName"],
            var_name=self.year_or_category_column_label,
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


class VariableCleanerMSA(
    VariableCleaner
):  # this class inherits functionalites of VariableCleaner, but works at the MSA level
    def __init__(
        self,
        variable_name: str,
        path_to_raw_csv: str,
        year_or_category_column_label: str = "Year",
    ):
        super().__init__(variable_name, path_to_raw_csv, year_or_category_column_label)
        self.folder = "MSA_level"
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


def weighted_mean(group, column):
    values = group[column]
    weights = group["Total population"]

    not_nan_indices = ~np.isnan(values)

    if np.any(not_nan_indices) and np.sum(weights[not_nan_indices]) != 0:
        weighted_values = values[not_nan_indices] * weights[not_nan_indices]
        return np.sum(weighted_values) / np.sum(weights[not_nan_indices])
    else:
        return np.nan


def communities_tracts_to_counties(
    data, list_variables
) -> pd.DataFrame:  # using the weighted mean function for total population
    all_results = pd.DataFrame()

    for variable in list_variables:
        weighted_avg = (
            data.groupby("GeoFIPS").apply(weighted_mean, column=variable).reset_index()
        )
        weighted_avg.columns = ["GeoFIPS", variable]

        nan_counties = (
            data.groupby("GeoFIPS")
            .apply(lambda x: all(np.isnan(x[variable])))
            .reset_index()
        )
        nan_counties.columns = ["GeoFIPS", "all_nan"]

        result_df = pd.merge(weighted_avg, nan_counties, on="GeoFIPS")
        result_df.loc[result_df["all_nan"], variable] = np.nan

        result_df = result_df.drop(columns=["all_nan"])

        if "GeoFIPS" not in all_results.columns:
            all_results = result_df.copy()
        else:
            all_results = pd.merge(all_results, result_df, on="GeoFIPS", how="left")

    return all_results


class VariableCleanerCT(
    VariableCleanerMSA
):  # this class inherits functionalites of two previous classes, but works at the Census Tract level
    def __init__(
        self,
        variable_name: str,
        path_to_raw_csv: str,
        time_interval: str,  # pre2020 or post2020
        year_or_category_column_label: str = "Year",
    ):
        super().__init__(variable_name, path_to_raw_csv, year_or_category_column_label)
        self.time_interval = time_interval
        self.folder = "Census_tract_level"
        self.census_tracts = None
        self.ct_list_file = None
        self.exclusions_file = None

    def clean_variable(self):
        self.load_raw_csv()
        self.drop_nans()
        self.load_census_tracts()
        self.check_exclusions()
        self.process_data()
        # TODO self.check_exclusions('CT') functionality might be usefull in the future
        self.save_csv_files(self.folder)

    def load_census_tracts(self):
        if self.time_interval == "pre2020":
            self.ct_list_file = f"{self.root}/data/raw/CT_list_pre2020.csv"
            self.census_tracts = pd.read_csv(self.ct_list_file)

            self.exclusions_file = f"{self.root}/data/raw/exclusions_ct_pre2020.csv"

        elif self.time_interval == "post2020":
            self.ct_list_file = f"{self.root}/data/raw/CT_list_post2020.csv"
            self.census_tracts = pd.read_csv(self.ct_list_file)

            self.exclusions_file = f"{self.root}/data/raw/exclusions_ct_post2020.csv"

    def add_new_exclusions(self, common_fips):
        new_exclusions = np.setdiff1d(
            self.census_tracts["GeoFIPS"].unique(), self.variable_df["GeoFIPS"].unique()
        )

        print(f"Adding new exclusions to {self.time_interval}: " + str(new_exclusions))

        # Append to exclusions file if it exists, otherwise create a new file
        try:
            existing_exclusions = pd.read_csv(self.exclusions_file)
            new_exclusions_df = pd.DataFrame(
                {
                    "dataset": [self.variable_name] * len(new_exclusions),
                    "exclusions": new_exclusions,
                }
            )
            updated_exclusions = (
                pd.concat([existing_exclusions, new_exclusions_df])
                .drop_duplicates()
                .reset_index(drop=True)
            )
        except FileNotFoundError:
            new_exclusions_df = pd.DataFrame(
                {
                    "dataset": [self.variable_name] * len(new_exclusions),
                    "exclusions": new_exclusions,
                }
            )
            updated_exclusions = new_exclusions_df

        updated_exclusions.to_csv(self.exclusions_file, index=False)

        self.census_tracts = self.census_tracts[
            ~self.census_tracts["GeoFIPS"].isin(new_exclusions)
        ]
        self.census_tracts.to_csv(self.ct_list_file, index=False)
        print("Updated CT list saved.")

        self.variable_df = self.variable_df[
            self.variable_df["GeoFIPS"].isin(self.census_tracts["GeoFIPS"])
        ]
        print(f"{self.variable_name} data updated.")

    def check_exclusions(self):

        assert (
            self.variable_df["GeoFIPS"].dtype == "int64"
        ), f"Expected 'int64', but got {self.variable_df['GeoFIPS'].dtype}"

        common_fips = np.intersect1d(
            self.census_tracts["GeoFIPS"].unique(), self.variable_df["GeoFIPS"].unique()
        )
        if (
            len(
                np.setdiff1d(
                    self.census_tracts["GeoFIPS"].unique(),
                    self.variable_df["GeoFIPS"].unique(),
                )
            )
            > 0
        ):
            self.add_new_exclusions(common_fips)
        else:
            print("No new exclusions needed.")
            pre_size = self.variable_df.shape[0]
            self.variable_df = self.variable_df[
                self.variable_df["GeoFIPS"].isin(self.census_tracts["GeoFIPS"])
            ]
            print(
                f" Initial size of {pre_size} of {self.variable_name} reduced to {self.variable_df.shape[0]}"
            )

    def process_data(self):
        self.load_census_tracts()

        self.variable_df["GeoFIPS"] = self.variable_df["GeoFIPS"].astype(np.int64)
        assert (
            self.census_tracts["GeoFIPS"].nunique()
            == self.variable_df["GeoFIPS"].nunique()
        ), f'FIPS mismatch! {self.census_tracts["GeoFIPS"].nunique()} vs {self.variable_df["GeoFIPS"].nunique()}'
