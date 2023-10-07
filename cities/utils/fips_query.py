

import os
import sys
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from scipy.spatial import distance

from .data_grabber import DataGrabber


class FipsQuery:

    def __init__(self, fips, outcome_var = "gdp", feature_groups = [], weights = None, lag = 0, top = 5): 
        
        #TODO add weights rescaling to init
        #TODO with a non-trival example of feature groups
    

        assert outcome_var in ["gdp"], "outcome_var must be one of ['gdp']" #TODO expand to other outcome vars
        



        self.data = DataGrabber()
        self.repo_root = self.data.repo_root
        self.fips = fips
        self.lag = lag
        self.top = top
        self.outcome_var = outcome_var
        
        self.data.get_gdp_std_wide()

        self.name = self.data.gdp_std_wide['GeoName'][self.data.gdp_std_wide['GeoFIPS'] == self.fips].values[0]


    def find_euclidean_kins(self, top = 5): ##TODO_Nikodem add a test for this function
        
        assert self.lag >= 0 and self.lag < 6 and  isinstance(self.lag, int),  "lag must be  an iteger between 0 and 5"
        assert (self.top > 0 and isinstance(self.top, int) and 
                    self.top < self.data.gdp_std_wide.shape[0]), "top must be a positive integer"

        if self.outcome_var == "gdp":
            df = self.data.gdp_std_wide
            original_length = df.shape[0]

            #TODO add other features here
            
            my_array = np.array(df[df['GeoFIPS'] == self.fips].values[0][2+self.lag:])
            other_df = df[df['GeoFIPS'] != self.fips].copy()
            if self.lag >0:
                other_df_cut = other_df.iloc[:, 2:-self.lag]
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

             
            
            


