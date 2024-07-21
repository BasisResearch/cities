from typing import Any, Dict, List, Optional

import pyro
import pyro.distributions as dist
import torch

from cities.modeling.zoning_models.units_causal_model import (get_n, categorical_contribution, 
                                                              continuous_contribution, add_linear_component, 
                                                              categorical_interaction_variable)


from cities.modeling.zoning_models.missingness_only_model import add_logistic_component




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
    ):
        if categorical_levels is None:
            categorical_levels = self.categorical_levels

        _N_categorical, _N_continuous, n = get_n(categorical, continuous)

        data_plate = pyro.plate("data", size=n, dim=-1)

        # #################
        # # register
        # #################
        with data_plate:

            year = pyro.sample(
                "year",
                dist.Categorical(torch.ones(len(categorical_levels["year"]))),
                obs=categorical["year"],
            )

            past_reform = pyro.sample(
                "past_reform",
                dist.Categorical(torch.ones(len(categorical_levels["past_reform"]))),
                obs=categorical["past_reform"],
            )

            total_value = pyro.sample(
                "total_value", dist.Normal(0, 1), obs=continuous["total_value"]
            )

            median_value = pyro.sample(
                "median_value", dist.Normal(0, 1), obs=continuous["median_value"]
            )


        # # ___________________________
        # # regression for housing units
        # # ___________________________
    
        housing_units_continuous_parents = {
            "total_value": total_value,
            "median_value": median_value,
        }

        housing_units_categorical_parents = {
            "year": year,
            "past_reform": past_reform,
        }

        housing_units = add_linear_component(
            child_name="housing_units",
            child_continuous_parents=housing_units_continuous_parents,
            child_categorical_parents=housing_units_categorical_parents,
            leeway= 0.9,
            data_plate=data_plate,
            observations=continuous["housing_units"],
        )

        return housing_units
        

        # applied_continuous_parents = {
        #     "value": value,
        # }
        # applied_categorical_parents = {
        #     "year": year,
        # }

        # applied = add_logistic_component(
        #     child_name="applied",
        #     child_continuous_parents=applied_continuous_parents,
        #     child_categorical_parents=applied_categorical_parents,
        #     leeway=11.57,
        #     data_plate=data_plate,
        #     observations= categorical["applied"],
        #     categorical_levels=categorical_levels,
        # )

        # return applied
