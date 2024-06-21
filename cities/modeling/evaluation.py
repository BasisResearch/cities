import torch
import pyro
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import os
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import ZoningDataset

from cities.modeling.simple_linear import SimpleLinear
from cities.modeling.svi_inference import run_svi_inference
from pyro.infer import Predictive
from chirho.robust.handlers.predictive import PredictiveModel
import matplotlib.pyplot as plt 
import seaborn as sns


root = find_repo_root()


def prep_data_for_test(train_size = 0.8):
    zoning_data_path =  os.path.join(root,"data/minneapolis/processed/zoning_dataset.pt")
    zoning_dataset_read = torch.load(zoning_data_path)

    train_size = int(train_size * len(zoning_dataset_read))
    test_size = len(zoning_dataset_read) - train_size

    train_dataset, test_dataset = random_split(zoning_dataset_read, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size = train_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size= test_size, shuffle=False)

    categorical_levels = zoning_dataset_read.categorical_levels

    return train_loader, test_loader, categorical_levels

def test_performance(model_class, kwarg_names, train_loader, test_loader, 
                     categorical_levels, n_steps=600, plot = True):

    assert all(item in kwarg_names.keys() for item in ["categorical", "continuous", "outcome"])
    assert kwarg_names["outcome"] not in kwarg_names["continuous"]

    train_data = next(iter(train_loader))
    train_data['outcome'] = train_data['continuous'][kwarg_names['outcome']]
    train_data['categorical'] = {key: val for key, val in train_data['categorical'].items() if key in kwarg_names['categorical']}
    train_data['continuous'] = {key: val for key, val in train_data['continuous'].items() if key in kwarg_names['continuous']}
   
    test_data = next(iter(test_loader))
    test_data['outcome'] = test_data['continuous'][kwarg_names['outcome']] 
    test_data['categorical'] = {key: val for key, val in test_data['categorical'].items() if key in kwarg_names['categorical']}
    test_data['continuous'] = {key: val for key, val in test_data['continuous'].items() if key in kwarg_names['continuous']}
    
    def apply_mask(data, mask):
        return {key: val[mask] for key, val in data.items()}

    mask = torch.ones(len(test_data['outcome']), dtype=torch.bool)
    for key, value in test_data['categorical'].items():
        mask = mask *  torch.isin(test_data['categorical'][key],(train_data['categorical'][key].unique()))
    # for name in test_data['categorical'].keys():
    #     test_data['categorical'][name] = test_data['categorical'][name][mask]
    # for name in test_data['continuous'].keys():
    #     test_data['continuous'][name] = test_data[mask]

    test_data['categorical'] = apply_mask(test_data['categorical'], mask)
    test_data['continuous'] = apply_mask(test_data['continuous'], mask)
    test_data['outcome'] = test_data['outcome'][mask]

    for key in test_data['categorical'].keys():
        assert test_data['categorical'][key].shape[0] == mask.sum()
    for key in test_data['continuous'].keys():
        assert test_data['continuous'][key].shape[0] == mask.sum()


    # raise error if sum(mask) < .5 * len(test_data['outcome'])
    if sum(mask) < .5 * len(test_data['outcome']):
        raise ValueError("Sampled test data has too many new categorical levels, consider decreasing train size")


    index_mapping =  {}
    for key in kwarg_names['categorical']:
        unique_values = train_data['categorical'][key].unique().tolist()
        value_to_index = {value: index for index, value in enumerate(unique_values)}
        index_mapping[key] = value_to_index
        train_data['categorical'][key] = torch.tensor([value_to_index[val.item()] for val in train_data['categorical'][key]], dtype=torch.long)
        test_data['categorical'][key] = torch.tensor([value_to_index[val.item()] for val in test_data['categorical'][key] if val.item() in value_to_index], dtype=torch.long)
    

    model = model_class(**train_data, categorical_levels=categorical_levels)

    guide = run_svi_inference(
    model, n_steps=n_steps, lr=0.01, verbose=True, **train_data
    )

    predictive = Predictive(
        model, guide=guide, num_samples = 1000
    )   

    samples = predictive(categorical = test_data['categorical'], 
                     continuous = test_data['continuous'],
                     outcome = None, )
    
    print(samples)

    print(train_data['categorical']['neighborhood_id'].unique())
    print(test_data['categorical']['neighborhood_id'].unique())
    # with pyro.plate("sampling", size = 1000, dim = -10):
    #     samples = predictive(categorical = test_data['categorical'], 
    #                      continuous = test_data['continuous'],
    #                      outcome = None).squeeze()[-1:]
        
    # predicted_means = samples.mean(axis=1)[-1]
    # lower_quantile = torch.quantile(samples, 0.05, dim=1)[-1]
    # upper_quantile = torch.quantile(samples, 0.95, dim=1)[-1]

    # if plot:
        
    #     true_outcome_np = test_data['outcome'].detach().numpy()
    #     predicted_means_np = predicted_means.detach().numpy()
    #     #lower_quantile_np = lower_quantile.detach().numpy()
    #     #upper_quantile_np = upper_quantile.detach().numpy()

    #     plt.scatter(true_outcome_np, predicted_means_np)
    #     #plt.errorbar(true_outcome_np, predicted_means_np, yerr=[predicted_means_np - lower_quantile_np, upper_quantile_np - predicted_means_np], fmt='o')
    #     plt.xlabel("true outcome")
    #     plt.ylabel("predicted outcome")
    #     plt.title("predicted vs true outcome")
    #     plt.show()

    #predicted_means, lower_quantile, upper_quantile


