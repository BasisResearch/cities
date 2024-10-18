from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


# works when run as a function
# within pytest leads to
# 'fixture `model _class` not found


# def test_evaluate():
#     train_loader, test_loader, categorical_levels = prep_data_for_test(train_size=0.8)

#     kwarg_names = {
#         "categorical": ["limit_id", "neighborhood_id"],
#         "continuous": {"parcel_area"},
#         "outcome": "housing_units",
#     }

#     test_performance(
#         SimpleLinear,
#         kwarg_names,
#         train_loader,
#         test_loader,
#         categorical_levels,
#         n_steps=10,
#         plot=False,
#     )
