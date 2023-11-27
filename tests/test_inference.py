from cities.modeling.model_interactions import InteractionsModel


# TODO this needs to be parametrized by outcome datasets,
# intervention datasets and forward shifts
def test_InteractionsModel():
    model = InteractionsModel(
        outcome_dataset="unemployment_rate",
        intervention_dataset="spending_commerce",
        intervention_variable="total_obligated_amount",
        forward_shift=2,
        num_iterations=10,
    )
    model.train_interactions_model()

    assert model.guide is not None
    assert model.model_args is not None
    assert model.model_conditioned is not None
