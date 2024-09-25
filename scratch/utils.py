import torch
from copy import deepcopy


def nonify_dict_(d):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = None
        elif isinstance(v, dict):
            d[k] = nonify_dict_(v)
    return d


def nonify_dict(d):
    d = deepcopy(d)
    return nonify_dict_(d)


SUBSET_SITE_NAME_MAP = {
    "white": "white_original",
    "segregation": "segregation_original",
    "limit": "mean_limit_original",
    "distance": "median_distance",
}


def map_subset_onto_obs(subset, site_names):
    obs = dict()

    for name in site_names:
        subset_name = SUBSET_SITE_NAME_MAP.get(name, name)
        for k, inner_subset_dict in subset.items():
            if k == "outcome":
                continue
            if subset_name in inner_subset_dict:
                obs[name] = inner_subset_dict[subset_name]
                break

    assert obs.keys() == set(site_names), f"Missing keys: {set(site_names) - obs.keys()}"
    return obs
