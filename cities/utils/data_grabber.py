import os
import sys
import pandas as pd

class DataGrabber:
    def __init__(self):
        self.repo_root = self.find_repo_root()
        sys.path.insert(0, self.repo_root)


    def find_repo_root(self):
            current_dir = os.getcwd()
            while True:
                marker_file_path = os.path.join(current_dir, '.gitignore') 
                if os.path.isfile(marker_file_path):
                    return current_dir 
            
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    break
                current_dir = parent_dir

    def get_gdp_wide(self):
        file_path = os.path.join(self.repo_root, "data/processed/gdp_wide.csv")
        self.gdp_wide = pd.read_csv(file_path)

    def get_gdp_std_wide(self):
        file_path = os.path.join(self.repo_root, "data/processed/gdp_std_wide.csv")
        self.gdp_std_wide = pd.read_csv(file_path)

    def get_gdp_long(self):
        file_path = os.path.join(self.repo_root, "data/processed/gdp_long.csv")
        self.gdp_long = pd.read_csv(file_path)

    def get_gdp_std_long(self):
        file_path = os.path.join(self.repo_root, "data/processed/gdp_std_long.csv")
        self.gdp_std_long = pd.read_csv(file_path) 
