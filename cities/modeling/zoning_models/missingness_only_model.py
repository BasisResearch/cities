from typing import Any, Dict, List, Optional

import pyro
import pyro.distributions as dist
import torch

from cities.modeling.zoning_models.units_causal_model import (get_n, categorical_contribution, 
                                                              continuous_contribution, add_linear_component, 
                                                              categorical_interaction_variable)



class MissingnessOnlyModel(pyro.nn.PyroModule):
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

        #################
        # register
        #################
        with data_plate:

            year = pyro.sample(
                "year",
                dist.Categorical(torch.ones(len(categorical_levels["year"]))),
                obs=categorical["year"],
            )

            parcel_area = pyro.sample(
                "parcel_area", dist.Normal(0, 1), obs=continuous["parcel_area"]
            )

            # month = pyro.sample(
            #     "month",
            #     dist.Categorical(torch.ones(len(categorical_levels["month"]))),
            #     obs=categorical["month"],
            # )

            # zone_id = pyro.sample(
            #     "zone_id",
            #     dist.Categorical(torch.ones(len(categorical_levels["zone_id"]))),
            #     obs=categorical["zone_id"],
            # )

            # neighborhood_id = pyro.sample(
            #     "neighborhood_id",
            #     dist.Categorical(
            #         torch.ones(len(categorical_levels["neighborhood_id"]))
            #     ),
            #     obs=categorical["neighborhood_id"],
            # )

            # ward_id = pyro.sample(
            #     "ward_id",
            #     dist.Categorical(torch.ones(len(categorical_levels["ward_id"]))),
            #     obs=categorical["ward_id"],
            # )

            # past_reform = pyro.sample(
            #     "past_reform", dist.Normal(0, 1), obs=categorical["past_reform"]
            # )

        # ___________________________
        # logistic regression for applied
        # ___________________________
    
        

        applied_continuous_parents = {
            "parcel_area": parcel_area,
        }
        applied_categorical_parents = {
            "year": year,
        }

        applied = add_linear_component(
            child_name="applied",
            child_continuous_parents=applied_continuous_parents,
            child_categorical_parents=applied_categorical_parents,
            leeway=leeway,
            data_plate=data_plate,
            observations=outcome,
            categorical_levels=categorical_levels,
        )

        return applied
