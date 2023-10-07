

import os
import sys
import pandas as pd
import numpy as np

from scipy.spatial import distance

from .data_grabber import DataGrabber


class FipsQuery:

    def __init__(self, fips, outcome_var = "gdp", feature_groups = [], weights = None): 
        
        #TODO add weights rescaling to init
        #TODO with a non-trival example of feature groups
    

        assert outcome_var in ["gdp"], "outcome_var must be one of ['gdp']" #TODO expand to other outcome vars
        



        self.data = DataGrabber()
        self.repo_root = self.data.repo_root
        self.fips = fips
        self.outcome_var = outcome_var
        
        self.data.get_gdp_std_wide()

        self.name = self.data.gdp_std_wide['GeoName'][self.data.gdp_std_wide['GeoFIPS'] == self.fips].values[0]


    def find_euclidean_kins(self,  lag = 0, top = 5): ##TODO_Nikodem add a test for this function
        
        assert lag >= 0 and lag < 6 and isinstance(lag, int), "lag must be an integer between 0 and 5"
        assert top > 0 and isinstance(top, int) and top < self.data.gdp_std_wide.shape[0], "top must be a positive integer"

        if self.outcome_var == "gdp":
            df = self.data.gdp_std_wide
            original_length = df.shape[0]

            my_array = np.array(df[df['GeoFIPS'] == self.fips].values[0][2+lag:])
            other_df = df[df['GeoFIPS'] != self.fips].copy()
            if lag >0:
                other_df_cut = other_df.iloc[:, 2:-lag]
            else:
                other_df_cut = other_df.iloc[:, 2:]
            other_arrays = np.array(other_df_cut.values)
            
            assert other_arrays.shape[0] + 1 == original_length, "Dataset sizes don't match"
            assert other_arrays.shape[1] == my_array.shape[0], "Lengths don't match"

            distances = []
            for vector in other_arrays:
                distances.append(distance.euclidean(my_array, vector, w = None))

            assert len(distances) == other_arrays.shape[0], "Something went wrong"

            other_df[f'distance to {self.fips}'] = distances

            other_df.sort_values(by=other_df.columns[-1], inplace=True)

            self.euclidean_kins = other_df 


