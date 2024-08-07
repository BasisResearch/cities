from typing import Any, List, Optional, Dict

import pyro
import pyro.distributions as dist
import torch

from cities.modeling.zoning_models.units_causal_model import (get_n, categorical_contribution, 
                                                              continuous_contribution, add_linear_component, 
                                                              categorical_interaction_variable)


from cities.modeling.zoning_models.missingness_only_model import add_logistic_component




def add_ratio_component(
    child_name: "str",
    child_continuous_parents,
    child_categorical_parents,
    leeway,  
    data_plate,
    observations=None,
    categorical_levels=None,
):


    continuous_contribution_to_child = continuous_contribution(
        child_continuous_parents, child_name, leeway
    )

    categorical_contribution_to_child = categorical_contribution(
        child_categorical_parents,
        child_name,
        leeway,
        categorical_levels=categorical_levels,
    )

    sigma_child = pyro.sample(
        f"sigma_{child_name}", dist.Exponential(40.0)
    ) 

    with data_plate:

        mean_prediction_child = pyro.deterministic(  # type: ignore
            f"mean_outcome_prediction_{child_name}",
            categorical_contribution_to_child + continuous_contribution_to_child,
            event_dim=0,
            )
                
        child_probs = pyro.deterministic(f"child_probs_{child_name}_{child_name}", 
                                         torch.sigmoid(mean_prediction_child),
                                         event_dim=0,)
        
        child_observed = pyro.sample(child_name, 
        dist.Normal(child_probs, sigma_child),
        obs=observations)


    return child_observed


def add_poisson_component(
    child_name: str,
    child_continuous_parents: Dict[str, torch.Tensor],
    child_categorical_parents: Dict[str, torch.Tensor],
    leeway: float,
    data_plate,
    observations: Optional[torch.Tensor] = None,
    categorical_levels: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:

    continuous_contribution_to_child = continuous_contribution(
        child_continuous_parents, child_name, leeway
    )

    categorical_contribution_to_child = categorical_contribution(
        child_categorical_parents,
        child_name,
        leeway,
        categorical_levels=categorical_levels,
    )

    with data_plate:

        mean_prediction_child = pyro.deterministic(
            f"mean_outcome_prediction_{child_name}",
            torch.exp(categorical_contribution_to_child + continuous_contribution_to_child),
            event_dim=0,
        )
        
        child_observed = pyro.sample(
            child_name,
            dist.Poisson(mean_prediction_child),
            obs=observations
        )

    return child_observed







class TractsModelNoRatios(pyro.nn.PyroModule):
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

            distance = pyro.sample("distance", dist.Normal(0, 1),
                                    obs=continuous["median_distance"])


            # past_reform = pyro.sample(
            #     "past_reform",
            #     dist.Categorical(torch.ones(len(categorical_levels["past_reform"]))),
            #     obs=categorical["past_reform"],
            # )


        ## ___________________________
        ## regression for white
        ## ___________________________

        white_continuous_parents = {
            "distance": distance,
        }

        white_categorical_parents = {
            "year": year,
        }

        white = add_linear_component(
            child_name="white",
            child_continuous_parents=white_continuous_parents,
            child_categorical_parents=white_categorical_parents,
            leeway=0.9,
            data_plate=data_plate,
            observations=continuous["white"],
        )

        ## ___________________________
        ## regression for segregation
        ## ___________________________

        segregation_continuous_parents = {
            "distance": distance,
            "white": white,
        }

        segregation_categorical_parents = {
            "year": year,
        }

        segregation = add_linear_component(
            child_name="segregation",
            child_continuous_parents=segregation_continuous_parents,
            child_categorical_parents=segregation_categorical_parents,
            leeway=0.9,
            data_plate=data_plate,
            observations=continuous["segregation"],
        )

        ## ___________________________
        ## regression for income
        ## ___________________________

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


        # #_____________________________
        # # regression for limit
        # #_____________________________
            

        limit_continuous_parents = {
            "distance": distance,
        }

        limit_categorical_parents = {
            "year": year,
        }

        limit = add_linear_component(
            child_name="limit",
            child_continuous_parents=limit_continuous_parents,
            child_categorical_parents=limit_categorical_parents,
            leeway=0.9,
            data_plate=data_plate,
            observations=continuous["mean_limit"],
        )

        # # _____________________________
        # # regression for median value
        # # _____________________________

        value_continuous_parents = {
            "distance": distance, "limit": limit,
            "income": income, "white": white,
            "segregation": segregation

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

        # # ___________________________
        # # regression for housing units
        # # ___________________________
    
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
            leeway= 0.9,
            data_plate=data_plate,
            observations=continuous["housing_units"],
        )

        return housing_units
        




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

            distance = pyro.sample("distance", dist.Normal(0, 1),
                                    obs=continuous["median_distance"])


            # past_reform = pyro.sample(
            #     "past_reform",
            #     dist.Categorical(torch.ones(len(categorical_levels["past_reform"]))),
            #     obs=categorical["past_reform"],
            # )


        ## ___________________________
        ## regression for white
        ## ___________________________

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

        ## ___________________________
        ## regression for segregation
        ## ___________________________

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

        ## ___________________________
        ## regression for income
        ## ___________________________

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


        # #_____________________________
        # # regression for limit
        # #_____________________________
            

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

        # # _____________________________
        # # regression for median value
        # # _____________________________

        value_continuous_parents = {
            "distance": distance, "limit": limit,
            "income": income, "white": white,
            "segregation": segregation

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




        # # ___________________________
        # # regression for housing units
        # # ___________________________
    
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
            leeway= 0.9,
            data_plate=data_plate,
            observations=continuous["housing_units"],
        )

        return housing_units
        



class TractsModelPoisson(pyro.nn.PyroModule):
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

            distance = pyro.sample("distance", dist.Normal(0, 1),
                                    obs=continuous["median_distance"])


            # past_reform = pyro.sample(
            #     "past_reform",
            #     dist.Categorical(torch.ones(len(categorical_levels["past_reform"]))),
            #     obs=categorical["past_reform"],
            # )


        ## ___________________________
        ## regression for white
        ## ___________________________

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

        ## ___________________________
        ## regression for segregation
        ## ___________________________

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

        ## ___________________________
        ## regression for income
        ## ___________________________

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


        # #_____________________________
        # # regression for limit
        # #_____________________________
            

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

        # # _____________________________
        # # regression for median value
        # # _____________________________

        value_continuous_parents = {
            "distance": distance, "limit": limit,
            "income": income, "white": white,
            "segregation": segregation

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




        # # ___________________________
        # # regression for housing units
        # # ___________________________
    
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

        housing_units = add_poisson_component(
            child_name="housing_units_original",
            child_continuous_parents=housing_units_continuous_parents,
            child_categorical_parents=housing_units_categorical_parents,
            leeway= 11.57,
            data_plate=data_plate,
            observations=continuous["housing_units_original"],
        )

        return housing_units
        





