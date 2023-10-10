import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  
sys.path.insert(0, parent_dir)


import pandas as pd
import numpy as np

import plotly.graph_objects as go
from scipy.spatial import distance
from cities.utils.data_grabber import DataGrabber
from cities.utils.similarity_utils import slice_with_lag




class FipsQuery:

    def __init__(self, fips, outcome_var = "gdp", feature_groups = [], weights = None, lag = 0, top = 5): 
        
        #TODO add weights rescaling to init
        #TODO with a non-trival example of feature groups
    
        assert outcome_var in ["gdp", "population"], "outcome_var must be one of ['gdp', 'population']"
        assert outcome_var not in feature_groups, "Outcome_var cannot be at the same time in background variables!"
        #TODO keep expanding to other outcome vars
       
        self.data = DataGrabber()
        self.repo_root = self.data.repo_root
        self.fips = fips
        self.lag = lag
        self.top = top
        self.outcome_var = outcome_var
        self.weights = weights
        
        self.data.get_gdp_std_wide()
        self.name = self.data.gdp_std_wide['GeoName'][self.data.gdp_std_wide['GeoFIPS'] == self.fips].values[0]

        assert self.lag >= 0 and self.lag < 6 and  isinstance(self.lag, int),  "lag must be  an iteger between 0 and 5"
        assert (self.top > 0 and isinstance(self.top, int) and 
                    self.top < self.data.gdp_std_wide.shape[0]), (
                "top must be a positive integer smaller than the number of locations in the dataset"
                    )
        
        self.feature_groups = feature_groups           
        for feature in self.feature_groups:
            self.data.get_feature_std_wide(feature)
        
        
            
            


    def find_euclidean_kins(self): ##TODO_Nikodem add a test for this function
        
        
        if self.outcome_var == "gdp":
            slices = slice_with_lag(self.data.gdp_std_wide, self.fips, self.lag)
        
        self.my_array = np.array(slices['my_array'])
        self.other_arrays = np.array(slices['other_arrays'])
        self.other_df = slices['other_df']
        
        #TODO add other features here
        #TODO will need to have the same fips codes in the same order in all other datasets
        #TODO for the feature addition to be introduced here to work smoothly
        
        distances = []
        for vector in self.other_arrays:
            distances.append(distance.euclidean(self.my_array, vector, w = self.weights))
        
        assert len(distances) == self.other_arrays.shape[0], "Something went wrong"

        self.other_df[f'distance to {self.fips}'] = distances
        
        self.euclidean_kins = self.other_df.sort_values(by=self.other_df.columns[-1])
        #TODO_Nikodem make sure this returns df with the original variable values, prior to normalization and rescaling


    def plot_kins(self):
        if self.outcome_var == "gdp":
            self.data.get_gdp_long()
            my_outcomes_long = self.data.gdp_long[self.data.gdp_long['GeoFIPS'] == self.fips].copy() 
          
            fips_top = self.euclidean_kins['GeoFIPS'].iloc[:self.top].values
            
            others_outcome_long = self.data.gdp_long[self.data.gdp_long['GeoFIPS'].isin(fips_top)]


            fig = go.Figure()
            fig.add_trace(go.Scatter(x=my_outcomes_long['Year'], y=my_outcomes_long['Value'],
                                      mode='lines', name=my_outcomes_long['GeoName'].iloc[0],
                                      line=dict(color='darkred', width=3),
                                      text=my_outcomes_long['GeoName'].iloc[0], 
                                      textposition='top right'
                                      ))

            #TODO_Nikodem add more shades and test on various settings of top
            shades_of_grey = ['#333333', '#444444', '#555555', '#666666', '#777777'][:self.top]
            pastel_colors = ['#FFC0CB', '#A9A9A9', '#87CEFA', '#FFD700', '#98FB98'][:self.top]

            #R: not sure which look better

            for i, geoname in enumerate(others_outcome_long['GeoName'].unique()):
                subset = others_outcome_long[others_outcome_long['GeoName'] == geoname]
                #line_color = shades_of_grey[i % len(shades_of_grey)]
                line_color = pastel_colors[i % len(pastel_colors)]
                fig.add_trace(go.Scatter(x=subset['Year'] + self.lag, y=subset['Value'],
                                        mode='lines', name=subset['GeoName'].iloc[0],
                                        line_color=line_color,
                                        text=subset['GeoName'].iloc[0], 
                                        textposition='top right'
                                        ))

            if self.lag >0:
                fig.update_layout(
                    shapes=[
                        dict(
                            type='line',
                            x0=2021,
                            x1=2021,
                            y0=0,
                            y1=1,
                            xref='x',
                            yref='paper',
                            line=dict(color='darkgray', width=2)
                        )
                    ]
                )

                fig.add_annotation(
                       text=f'their year {2021 - self.lag}',
                        x=2021.,
                        y=1.05, 
                        xref='x',
                        yref='paper',
                        showarrow=False,
                        font=dict(color='darkgray')
                        )



            fig.update_layout(
                title=f'Top {self.top} locations whose GDP patterns up to year {2021-self.lag} are most similar to the current pattern of {self.name}', 
                xaxis_title='Year',
                yaxis_title='Chain-type quantity indexes for real GDP',
                legend=dict(title='GeoName'),
                template = "simple_white",
            )

            fig.show()


#TODO_Nikodem add population clustering and warning if a population is much different,
#especially if small

             
            
            


