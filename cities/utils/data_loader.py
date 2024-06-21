from torch.utils.data import Dataset
import torch

class ZoningDataset(Dataset):
    def __init__(self, categorical, continuous, standardization_dictionary=None, index_dictionary=None):
        self.categorical = categorical
        self.continuous = continuous

        if index_dictionary is None:
            self.index_dictionary = {
                "zoning_ordering" : ['downtown', 'blue_zone', 'yellow_zone', 'other_non_university'],
                "limit_ordering" : ['eliminated', 'reduced', 'full']
            }

        self.standardization_dictionary = standardization_dictionary

        categorical_levels = dict()
        if self.categorical:
            self.categorical_levels = dict()
            for name in self.categorical.keys():
                self.categorical_levels[name] = torch.unique(categorical[name])

    def __len__(self):
        return len(self.categorical['parcel_id'])
    
    def __getitem__(self, idx):
        cat_data = {key: val[idx] for key, val in self.categorical.items()}
        cont_data = {key: val[idx] for key, val in self.continuous.items()}
        return {'categorical': cat_data, 'continuous': cont_data,}
    