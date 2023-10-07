import os
import sys
import pandas as pd

class DataGrabber:
    def __init__(self):
        sys.path.insert(0, os.path.dirname(os.getcwd()))

    def get_gpd_wide(self):
        self.gpd_wide = pd.read_csv("../data/processed/gdp_wide.csv")
    
    def get_gpd_std_wide(self):
        self.gpd_std_wide =  pd.read_csv("../data/processed/gdp_std_wide.csv")

    def get_gpd_long(self):
        self.gpd_long = pd.read_csv("../data/processed/gdp_long.csv")

    def get_gpd_std_long(self):
        self.gpd_std_long = pd.read_csv("../data/processed/gdp_std_long.csv")