import pyro
import torch

from pyro import distributions as dist
from typing import Dict, Any, Optional



from cities.modeling.model_components import (
    get_n,
    get_categorical_levels,
    check_categorical_is_subset_of_levels,
    add_linear_component,
)


from cities.modeling.zoning_models.ts_model_components import (add_ar1_component_with_interactions, reshape_into_time_series)
import warnings




class TractsModelCumulativeAR1(pyro.nn.PyroModule):
    def __init__(
        self,
        data: Dict,
        categorical_levels: Optional[Dict[str, Any]] = None,
        leeway=0.9,
        housing_units_continuous_parents_names = [],
        housing_units_continuous_interaction_pairs=[],    
    ):
      

        super().__init__()

        self.housing_units_continuous_parents_names = housing_units_continuous_parents_names

        categorical = data["categorical"]
        continuous = data["continuous"]

        self.leeway = leeway
        self.housing_units_continuous_interaction_pairs = (
            housing_units_continuous_interaction_pairs
        )
        

        self.N_categorical, self.N_continuous, n = get_n(categorical, continuous)

        if self.N_categorical > 0 and categorical_levels is None:
            self.categorical_levels = get_categorical_levels(categorical)
        else:
            self.categorical_levels = categorical_levels

        self.outcome_reshaped = None
        self.continuous_parents_reshaped = None
        self.categorical_parents_reshaped = None

    def forward(
        self,
        data: Dict,
        leeway= 0.9,
        categorical_levels=None,
        n=None,
        force_ts_reshape = False,
    ):
        
        categorical = data["categorical"]
        continuous = data["continuous"]
        if 'init_cumulative_state' in data:
            init_state = data['init_cumulative_state']
        else:
            init_state = None

        
        if categorical_levels is not None:
            warnings.warn(
                "Passed categorical_levels will no longer override the levels passed to or computed during"
                " model initialization. The argument will be ignored."
            )

        categorical_levels = self.categorical_levels
        assert check_categorical_is_subset_of_levels(categorical, categorical_levels)

        if n is None:
            _, _, n = get_n(categorical, continuous)

        # get init state from data if data available but no specific init state passed
        if init_state is None and continuous['housing_units_cumulative'] is not None:

            init_state = continuous['housing_units_cumulative'][categorical['year'] == 0].unsqueeze(-1)

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

    

            ## temporary plug in of predictors meant to come from a causal model

            # median_value = pyro.sample(
            #     "median_value", dist.Normal(0, 1), obs=continuous["median_value"]
            # )
            
            # income = pyro.sample( 
            #     "income", dist.Normal(0, 1), obs=continuous["income"]
            # )

            white = pyro.sample(
                "white", dist.Normal(0, 1), obs=continuous["white_original"]
            )

            limit = pyro.sample(
                "limit", dist.Normal(0, 1), obs=continuous["mean_limit_original"]
            )

            segregation = pyro.sample(
                "segregation", dist.Normal(0, 1), obs=continuous["segregation_original"]
            )

            sqm = pyro.sample(
                "sqm", dist.Normal(0, 1), obs=continuous["parcel_sqm"]
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

        #  _____________________________
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


        ## now applying the ar1 component
        all_possible_housing_units_continuous_parents = {
            "median_value": median_value,
            "distance": distance,
            "income": income,
            "white": white,
            "limit": limit,
            "segregation": segregation,
            "sqm": sqm,
            "downtown_overlap": downtown_overlap,
            "university_overlap": university_overlap
        }

        housing_units_continuous_parents = {
            key: all_possible_housing_units_continuous_parents[key]
            for key in self.housing_units_continuous_parents_names
        }

        housing_units_categorical_parents = {
            "year": year,
        }

        housing_units_cumulative = add_ar1_component_with_interactions(self,
            series_idx = categorical["census_tract"],
            time_idx = categorical["year"],
            child_name="housing_units_cumulative",
            child_continuous_parents=housing_units_continuous_parents,
            child_categorical_parents=housing_units_categorical_parents,
            continous_interaction_pairs=self.housing_units_continuous_interaction_pairs,
            leeway=leeway,
            data_plate=data_plate,
            initial_observations = init_state,
            observations = continuous['housing_units_cumulative'],
            categorical_levels=self.categorical_levels,
            force_ts_reshape = force_ts_reshape
        )





