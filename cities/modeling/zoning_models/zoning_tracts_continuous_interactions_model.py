import warnings
from typing import Any, Dict, Optional

import pyro
import pyro.distributions as dist
import torch

from cities.modeling.model_components import (
    add_linear_component,
    add_linear_component_continuous_interactions,
    add_ratio_component,
    check_categorical_is_subset_of_levels,
    get_categorical_levels,
    get_n,
)


class TractsModelContinuousInteractions(pyro.nn.PyroModule):
    def __init__(
        self,
        categorical: Dict[str, torch.Tensor],
        continuous: Dict[str, torch.Tensor],
        outcome: Optional[
            torch.Tensor
        ] = None,  # init args kept for uniformity, consider deleting
        categorical_levels: Optional[Dict[str, Any]] = None,
        leeway=0.9,
    ):
        """

        :param categorical: dict of categorical data
        :param continuous: dict of continuous data
        :param outcome: outcome data  (unused, todo remove)
        :param categorical_levels: dict of unique categorical values. If this is not passed, it will be computed from
         the provided categorical data. Importantly, if categorical is a subset of the full dataset, this automated
         computation may omit categorical levels that are present in the full dataset but not in the subset.
        """
        super().__init__()

        self.leeway = leeway

        self.N_categorical, self.N_continuous, n = get_n(categorical, continuous)

        if self.N_categorical > 0 and categorical_levels is None:
            self.categorical_levels = get_categorical_levels(categorical)
        else:
            self.categorical_levels = categorical_levels

    def forward(
        self,
        categorical: Dict[str, torch.Tensor],
        continuous: Dict[str, torch.Tensor],
        outcome: Optional[torch.Tensor] = None,
        leeway=0.9,
        categorical_levels=None,
        n=None,
    ):
        if categorical_levels is not None:
            warnings.warn(
                "Passed categorical_levels will no longer override the levels passed to or computed during"
                " model initialization. The argument will be ignored."
            )

        categorical_levels = self.categorical_levels
        assert check_categorical_is_subset_of_levels(categorical, categorical_levels)

        if n is None:
            _, _, n = get_n(categorical, continuous)

        data_plate = pyro.plate("data", size=n, dim=-1)

        # _________
        # register
        # _________

        with data_plate:

            year = pyro.sample(
                "year",
                dist.Categorical(torch.ones(len(categorical_levels["year"]))),
                obs=categorical["year"],
            )

            distance = pyro.sample(
                "distance", dist.Normal(0, 1), obs=continuous["median_distance"]
            )

            downtown_overlap = pyro.sample(
                "downtown_overlap",
                dist.Normal(0, 1),
                obs=continuous["downtown_overlap"],
            )

            university_overlap = pyro.sample(
                "university_overlap",
                dist.Normal(0, 1),
                obs=continuous["university_overlap"],
            )

        # ______________________
        # regression for sqm
        # ______________________

        sqm_continuous_parents = {
            "distance": distance,
        }

        sqm_categorical_parents = {
            "year": year,
        }

        sqm = add_linear_component(
            child_name="sqm",
            child_continuous_parents=sqm_continuous_parents,
            child_categorical_parents=sqm_categorical_parents,
            leeway=0.5,
            data_plate=data_plate,
            observations=continuous["parcel_sqm"],
            categorical_levels=self.categorical_levels,
        )

        # _______________________
        # regression for limit
        # _______________________

        limit_continuous_parents = {
            "distance": distance,
        }

        limit_categorical_parents = {
            "year": year,
        }

        limit = add_ratio_component(
            child_name="limit",
            child_continuous_parents=limit_continuous_parents,
            child_categorical_parents=limit_categorical_parents,
            leeway=8,  # ,
            data_plate=data_plate,
            observations=continuous["mean_limit_original"],
            categorical_levels=self.categorical_levels,
        )

        # _____________________
        # regression for white
        # _____________________

        white_continuous_parents = {
            "distance": distance,
            "sqm": sqm,
            "limit": limit,
        }

        white_categorical_parents = {
            "year": year,
        }

        white = add_ratio_component(
            child_name="white",
            child_continuous_parents=white_continuous_parents,
            child_categorical_parents=white_categorical_parents,
            leeway=8,  # 11.57,
            data_plate=data_plate,
            observations=continuous["white_original"],
            categorical_levels=self.categorical_levels,
        )

        # ___________________________
        # regression for segregation
        # ___________________________

        segregation_continuous_parents = {
            "distance": distance,
            "white": white,
            "sqm": sqm,
            "limit": limit,
        }

        segregation_categorical_parents = {
            "year": year,
        }

        segregation = add_ratio_component(
            child_name="segregation",
            child_continuous_parents=segregation_continuous_parents,
            child_categorical_parents=segregation_categorical_parents,
            leeway=8,  # 11.57,
            data_plate=data_plate,
            observations=continuous["segregation_original"],
            categorical_levels=self.categorical_levels,
        )

        # ______________________
        # regression for income
        # ______________________

        income_continuous_parents = {
            "distance": distance,
            "white": white,
            "segregation": segregation,
            "sqm": sqm,
            "limit": limit,
        }

        income_categorical_parents = {
            "year": year,
        }

        income = add_linear_component(
            child_name="income",
            child_continuous_parents=income_continuous_parents,
            child_categorical_parents=income_categorical_parents,
            leeway=0.5,
            data_plate=data_plate,
            observations=continuous["income"],
            categorical_levels=self.categorical_levels,
        )

        # _____________________________
        # regression for median value
        # _____________________________

        value_continuous_parents = {
            "distance": distance,
            "income": income,
            "white": white,
            "segregation": segregation,
            "sqm": sqm,
            "limit": limit,
        }

        value_categorical_parents = {
            "year": year,
        }

        median_value = add_linear_component(
            child_name="median_value",
            child_continuous_parents=value_continuous_parents,
            child_categorical_parents=value_categorical_parents,
            leeway=0.5,
            data_plate=data_plate,
            observations=continuous["median_value"],
            categorical_levels=self.categorical_levels,
        )

        # ______________________________
        # regression for housing units
        # ______________________________

        housing_units_continuous_parents = {
            "median_value": median_value,
            "distance": distance,
            "income": income,
            "white": white,
            "limit": limit,
            "segregation": segregation,
            "sqm": sqm,
            "downtown_overlap": downtown_overlap,
            "university_overlap": university_overlap,
        }

        housing_units_categorical_parents = {
            "year": year,
        }

        housing_units_continuous_interaction_pairs = [
            ("university_overlap", "limit"),
        ]

        housing_units = add_linear_component_continuous_interactions(
            child_name="housing_units",
            child_continuous_parents=housing_units_continuous_parents,
            child_categorical_parents=housing_units_categorical_parents,
            continous_interaction_pairs=housing_units_continuous_interaction_pairs,
            leeway=0.5,
            data_plate=data_plate,
            observations=continuous["housing_units"],
            categorical_levels=self.categorical_levels,
        )

        return housing_units
