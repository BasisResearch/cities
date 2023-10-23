import pytest

from cities.queries.fips_query import FipsQuery
from cities.utils.data_grabber import DataGrabber

data = DataGrabber()


def test_fips_query_init():
    f1007 = FipsQuery(
        fips=1007,
        outcome_var="gdp",
        feature_groups_with_weights={"gdp": 4, "population":4},
        lag=0,
        top=8,
    )

    assert f1007.outcome_var == "gdp"
    assert f1007.feature_groups == ["gdp", "population"]
    assert list(f1007.data.std_wide.keys()) == ["gdp", "population"]

    assert f1007.data.std_wide["gdp"].shape[0] > 100
    assert f1007.data.std_wide["population"].shape[0] > 100


queries = [
    FipsQuery(42001, "gdp", lag=0, top=5, time_decay=1.06),
    FipsQuery(
        1007,
        outcome_var="gdp",
        feature_groups_with_weights= {"gdp": 4, "population": 4},
        lag=0,
        top=5,
        time_decay=1.03,
    ),
    FipsQuery(42001, "gdp", lag=2, top=5, time_decay=1.06),
    FipsQuery(
        20003,
        outcome_var="gdp",
        feature_groups_with_weights={"gdp": 4, "population": 4},
        lag=2,
        top=5,
        time_decay=1.03,
    ),
]


@pytest.mark.parametrize("query", queries)
def test_euclidean_kins_dont_die(query):
    f = query
    f.find_euclidean_kins()
    
