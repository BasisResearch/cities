import random
import time

from cities.queries.causal_insight import CausalInsight
from cities.utils.data_grabber import DataGrabber, list_interventions, list_outcomes

data = DataGrabber()
data.get_features_wide(["gdp"])
gdp = data.wide["gdp"]


interventions = list_interventions()
outcomes = list_outcomes()
fips_codes = gdp["GeoFIPS"].unique()
intervened_value = random.uniform(-1, 1)

# consider switching to random test
# downstream once everything is stable
# intervention = random.choice(interventions)
# outcome = random.choice(outcomes)
# intervened_value = random.uniform(-1, 1)
# fips = random.choice(fips_codes)
# def test_slim_random():

#     ci = CausalInsight(
#         outcome_dataset=outcome,
#         intervention_dataset=intervention,
#         num_samples=1000,
#     )

#     ci.get_tau_samples()
#     ci.get_fips_predictions(intervened_value=intervened_value, fips=fips)

#     assert len(ci.tensed_tau_samples[1]) == 1000
#     assert ci.predictions is not None


def test_slim_full():
    for intervention in interventions:
        for outcome in outcomes:
            fips = random.choice(fips_codes)

            ci = CausalInsight(
                outcome_dataset=outcome,
                intervention_dataset=intervention,
                num_samples=1000,
            )

            ci.get_tau_samples()

            ci.get_fips_predictions(intervened_value=intervened_value, fips=fips)

            assert len(ci.tensed_tau_samples[1]) == 1000
            assert ci.predictions is not None

a = time.time()
test_slim_full()
b = time.time()
print(b-a)
print((b-a)/60)
