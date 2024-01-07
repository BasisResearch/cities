import numpy as np

from cities.queries.causal_insight import CausalInsight

# TODO parametrize by other interventions and a few other fips, years and outcomes


def test_slider():
    fips = 42001
    outcome = "gdp"
    intervention = "spending_commerce"
    year = 2017

    ci = CausalInsight(
        outcome_dataset=outcome,
        intervention_dataset=intervention,
        num_samples=1000,
    )

    percent_calc = ci.slider_values_to_interventions(intervened_percent=0.5, year=year)

    ci.get_tau_samples()

    ci.get_fips_predictions(
        intervened_value=percent_calc["intervened_transformed"], fips=fips, year=year
    )

    assert ci.predictions.shape == (4, 5)

    assert np.allclose(
        ci.intervened_value_original, percent_calc["intervened_original"], rtol=0.01
    )
