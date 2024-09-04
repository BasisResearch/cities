from typing import Any, Dict, Optional

import pyro
import pyro.distributions as dist
import torch

from cities.modeling.model_components import (
    add_linear_component,
    add_ratio_component,
    get_n,
)


class TractsModel(pyro.nn.PyroModule):
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
        super().__init__()

        self.leeway = leeway

        self.N_categorical, self.N_continuous, n = get_n(categorical, continuous)

        # you might need and pass further the original
        #  categorical levels of the training data
        if self.N_categorical > 0 and categorical_levels is None:
            self.categorical_levels = dict()
            for name in categorical.keys():
                self.categorical_levels[name] = torch.unique(categorical[name])
        else:
            self.categorical_levels = categorical_levels  # type: ignore

    def forward(
        self,
        categorical: Dict[str, torch.Tensor],
        continuous: Dict[str, torch.Tensor],
        outcome: Optional[torch.Tensor] = None,
        categorical_levels: Optional[Dict[str, torch.Tensor]] = None,
        leeway=0.9,
        n=None
    ):
        if categorical_levels is None:
            categorical_levels = self.categorical_levels

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

        # _____________________
        # regression for white
        # _____________________

        white_continuous_parents = {
            "distance": distance,
        }

        white_categorical_parents = {
            "year": year,
        }

        white = add_ratio_component(
            child_name="white",
            child_continuous_parents=white_continuous_parents,
            child_categorical_parents=white_categorical_parents,
            leeway=11.57,
            data_plate=data_plate,
            observations=continuous["white_original"],
        )

        # ___________________________
        # regression for segregation
        # ___________________________

        segregation_continuous_parents = {
            "distance": distance,
            "white": white,
        }

        segregation_categorical_parents = {
            "year": year,
        }

        segregation = add_ratio_component(
            child_name="segregation",
            child_continuous_parents=segregation_continuous_parents,
            child_categorical_parents=segregation_categorical_parents,
            leeway=11.57,
            data_plate=data_plate,
            observations=continuous["segregation_original"],
        )

        # ______________________
        # regression for income
        # ______________________

        income_continuous_parents = {
            "distance": distance,
            "white": white,
            "segregation": segregation,
        }

        income_categorical_parents = {
            "year": year,
        }

        income = add_linear_component(
            child_name="income",
            child_continuous_parents=income_continuous_parents,
            child_categorical_parents=income_categorical_parents,
            leeway=0.9,
            data_plate=data_plate,
            observations=continuous["income"],
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
            leeway=11.57,
            data_plate=data_plate,
            observations=continuous["mean_limit_original"],
        )

        # _____________________________
        # regression for median value
        # _____________________________

        value_continuous_parents = {
            "distance": distance,
            "limit": limit,
            "income": income,
            "white": white,
            "segregation": segregation,
        }

        value_categorical_parents = {
            "year": year,
        }

        median_value = add_linear_component(
            child_name="median_value",
            child_continuous_parents=value_continuous_parents,
            child_categorical_parents=value_categorical_parents,
            leeway=0.9,
            data_plate=data_plate,
            observations=continuous["median_value"],
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
        }

        housing_units_categorical_parents = {
            "year": year,
        }

        housing_units = add_linear_component(
            child_name="housing_units",
            child_continuous_parents=housing_units_continuous_parents,
            child_categorical_parents=housing_units_categorical_parents,
            leeway=0.9,
            data_plate=data_plate,
            observations=continuous["housing_units"],
        )

        return housing_units
