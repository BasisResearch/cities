import os
import sys

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from scipy.spatial import distance
from cities.utils.data_grabber import DataGrabber
from cities.utils.similarity_utils import (slice_with_lag, compute_weight_array)

import matplotlib.pyplot as plt


class FipsQuery:

    def __init__(self, fips, outcome_var = "gdp", feature_groups = [], weights = {}, lag = 0, top = 5, time_decay = 1.08): 
        
    
        assert outcome_var in ["gdp", "population"], "outcome_var must be one of ['gdp', 'population']"
        assert outcome_var not in feature_groups, "Outcome_var cannot be at the same time in background variables!"
        #assert feature_groups == list(weights.keys()), "feature_groups and weights must correspond!."
        #TODO_Nikodem fix the above assertion to be useful
       
            
       
        self.data = DataGrabber()
        self.repo_root = self.data.repo_root
        self.fips = fips
        self.lag = lag
        self.top = top
        self.outcome_var = outcome_var
        self.time_decay = time_decay
        
        if len(feature_groups) > 0:
            assert len(weights) == len(feature_groups), "feature_groups and weights must correspond!"
        self.feature_groups = feature_groups
        self.weights = weights
        
    
        self.all_features = [outcome_var] + feature_groups
        all_features_with_gdp = self.all_features
        if "gdp" not in self.all_features:
            all_features_with_gdp.append("gdp")
        
        # we want the gdp added to be the source of truth for indexing
        # but we will use all_features in weighting
        # where we might not have gdp
    
        self.data.get_features_std_wide(all_features_with_gdp)
        self.data.get_features_wide(all_features_with_gdp)
        
        #TODO_Nikodem: here you need to implement testing if the features are a time series, and 
        #TODO_Nikodem: dropping columns that are excluded by `how_far_back`
        
        assert fips in self.data.std_wide['gdp']['GeoFIPS'].values , "FIPS not found in the data set."
        self.name = self.data.std_wide["gdp"]['GeoName'][self.data.std_wide["gdp"]['GeoFIPS'] == self.fips].values[0]


        assert self.lag >= 0 and self.lag < 6 and  isinstance(self.lag, int),  "lag must be  an iteger between 0 and 5"
        assert (self.top > 0 and isinstance(self.top, int) and 
                    self.top < self.data.std_wide[self.outcome_var].shape[0]), (
                "top must be a positive integer smaller than the number of locations in the dataset"
                    )
        

    def find_euclidean_kins(self): ##TODO_Nikodem add a test for this function
        
        
        self.outcome_slices = slice_with_lag(self.data.std_wide[self.outcome_var],
                                              self.fips, self.lag)
 
    
        self.my_array = np.array(self.outcome_slices['my_array'])
        self.other_arrays = np.array(self.outcome_slices['other_arrays'])
    
        assert self.my_array.shape[1] == self.other_arrays.shape[1]
        
        self.my_df = self.data.wide[self.outcome_var][self.data.wide[self.outcome_var]['GeoFIPS'] == self.fips].copy()
        
        self.other_df = self.outcome_slices['other_df']
        self.other_df = self.data.wide[self.outcome_var][self.data.wide[self.outcome_var]['GeoFIPS'] != self.fips].copy()
        

        # add data on other features listed to the arrays
        # prior to distance computation
        my_features_arrays = np.array([])
        others_features_arrays = np.array([])
        for feature in self.feature_groups:
            _extracted_df = self.data.wide[feature].copy()
            _extracted_my_df = _extracted_df[_extracted_df['GeoFIPS'] == self.fips]
            _extracted_other_df = _extracted_df[_extracted_df['GeoFIPS'] != self.fips]

            before_shape =  self.other_df.shape

            assert (self.other_df['GeoFIPS'].unique() == 
                    _extracted_other_df['GeoFIPS'].unique()).all(), "FIPS are missing"
            
            assert (self.other_df['GeoFIPS']== _extracted_other_df['GeoFIPS']).all(), "FIPS are misaligned"

            _extracted_other_df.columns = [f'{col}_{feature}' if col not
                                            in ['GeoFIPS', 'GeoName'] else col for col in
                                              _extracted_other_df.columns]
            

            _extracted_my_df.columns = [f'{col}_{feature}' if col not
                                            in ['GeoFIPS', 'GeoName'] else col for col in
                                              _extracted_my_df.columns]
            
            self.my_df = pd.concat((self.my_df, _extracted_my_df.iloc[:,2:]), axis = 1)
            self.other_df = pd.concat((self.other_df, _extracted_other_df.iloc[:,2:]), axis = 1)
            
            after_shape =  self.other_df.shape

            assert before_shape[0] == after_shape[0], "Feature merging went wrong!"


            _extracted_df_std = self.data.std_wide[feature].copy()
            _extracted_other_array = np.array(_extracted_df_std[_extracted_df_std['GeoFIPS'] != self.fips].iloc[:, 2:])
            _extracted_my_array = np.array(_extracted_df_std[_extracted_df_std['GeoFIPS'] == self.fips].iloc[:, 2:])
            

            if my_features_arrays.size == 0:
                my_features_arrays = _extracted_my_array
            else:
                my_features_arrays = np.hstack((my_features_arrays, _extracted_my_array))
            
            if others_features_arrays.size == 0:
                others_features_arrays = _extracted_other_array
            else:
                others_features_arrays = np.hstack((others_features_arrays, _extracted_other_array))
        
        if self.feature_groups:
            self.my_array = np.hstack((self.my_array, my_features_arrays))        
            self.other_arrays = np.hstack((self.other_arrays, others_features_arrays))
         

        compute_weight_array(self, self.time_decay)

        diff =   self.all_weights.shape[0] -  self.other_arrays.shape[1]
        self.all_weights = self.all_weights[diff:]

        assert self.other_arrays.shape[1] == self.all_weights.shape[0], "Weights and arrays are misaligned" 

        distances = []
        for vector in self.other_arrays:
            distances.append(distance.euclidean(np.squeeze(self.my_array), vector, w = self.all_weights))
        
        count = sum([1 for distance in distances if distance == 0])
       

        assert len(distances) == self.other_arrays.shape[0], "Distances and arrays are misaligned"
        assert len(distances) == self.other_df.shape[0], "Distances and df are misaligned"

        self.other_df[f'distance to {self.fips}'] = distances
        count_zeros = (self.other_df[f'distance to {self.fips}'] == 0 ).sum()

        assert count_zeros == count, "f{count_zeros} zeros in alien distances!"

        self.other_df.sort_values(by=self.other_df.columns[-1], inplace=True)

        self.my_df[f'distance to {self.fips}' ] = 0

        self.euclidean_kins = pd.concat((self.my_df, self.other_df), axis = 0)

        
        

#     def plot_kins(self):
#         if self.outcome_var == "gdp":
#             self.data.get_gdp_long()
#             my_outcomes_long = self.data.gdp_long[self.data.gdp_long['GeoFIPS'] == self.fips].copy() 
          
#             fips_top = self.euclidean_kins['GeoFIPS'].iloc[:self.top].values
            
#             others_outcome_long = self.data.gdp_long[self.data.gdp_long['GeoFIPS'].isin(fips_top)]


#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=my_outcomes_long['Year'], y=my_outcomes_long['Value'],
#                                       mode='lines', name=my_outcomes_long['GeoName'].iloc[0],
#                                       line=dict(color='darkred', width=3),
#                                       text=my_outcomes_long['GeoName'].iloc[0], 
#                                       textposition='top right'
#                                       ))

#             #TODO_Nikodem add more shades and test on various settings of top
#             shades_of_grey = ['#333333', '#444444', '#555555', '#666666', '#777777'][:self.top]
#             pastel_colors = ['#FFC0CB', '#A9A9A9', '#87CEFA', '#FFD700', '#98FB98'][:self.top]

#             #R: not sure which look better

#             for i, geoname in enumerate(others_outcome_long['GeoName'].unique()):
#                 subset = others_outcome_long[others_outcome_long['GeoName'] == geoname]
#                 #line_color = shades_of_grey[i % len(shades_of_grey)]
#                 line_color = pastel_colors[i % len(pastel_colors)]
#                 fig.add_trace(go.Scatter(x=subset['Year'] + self.lag, y=subset['Value'],
#                                         mode='lines', name=subset['GeoName'].iloc[0],
#                                         line_color=line_color,
#                                         text=subset['GeoName'].iloc[0], 
#                                         textposition='top right'
#                                         ))

#             if self.lag >0:
#                 fig.update_layout(
#                     shapes=[
#                         dict(
#                             type='line',
#                             x0=2021,
#                             x1=2021,
#                             y0=0,
#                             y1=1,
#                             xref='x',
#                             yref='paper',
#                             line=dict(color='darkgray', width=2)
#                         )
#                     ]
#                 )

#                 fig.add_annotation(
#                        text=f'their year {2021 - self.lag}',
#                         x=2021.,
#                         y=1.05, 
#                         xref='x',
#                         yref='paper',
#                         showarrow=False,
#                         font=dict(color='darkgray')
#                         )



#             fig.update_layout(
#                 title=f'Top {self.top} locations whose GDP patterns up to year {2021-self.lag} are most similar to the current pattern of {self.name}', 
#                 xaxis_title='Year',
#                 yaxis_title='Chain-type quantity indexes for real GDP',
#                 legend=dict(title='GeoName'),
#                 template = "simple_white",
#             )

#             fig.show()


# #TODO_Nikodem add population clustering and warning if a population is much different,
# #especially if small

             
            
            


