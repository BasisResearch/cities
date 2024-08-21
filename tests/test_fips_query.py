import pytest

from cities.queries.fips_query import CTFipsQuery, FipsQuery, MSAFipsQuery
from cities.utils.data_grabber import DataGrabber

data = DataGrabber()


def test_fips_query_init():
    f1007 = FipsQuery(
        fips=1007,
        outcome_var="gdp",
        feature_groups_with_weights={"gdp": 4, "population": 4},
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
        feature_groups_with_weights={"gdp": 4, "population": 4},
        lag=0,
        top=5,
        time_decay=1.03,
    ),
    FipsQuery(
        1007,
        feature_groups_with_weights={"gdp": 4, "population": 4},
        lag=0,
        top=5,
        time_decay=1.03,
    ),
    FipsQuery(
        1007,
        outcome_var="gdp",
        feature_groups_with_weights={"gdp": 0, "population": 4},
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


# MSA level


def test_fips_query_MSA_init():
    f1007 = MSAFipsQuery(
        fips=10780,
        outcome_var="gdp_ma",
        feature_groups_with_weights={"gdp_ma": 4, "population_ma": 4},
        lag=0,
        top=8,
    )

    assert f1007.outcome_var == "gdp_ma"
    assert f1007.feature_groups == ["gdp_ma", "population_ma"]
    assert list(f1007.data.std_wide.keys()) == ["gdp_ma", "population_ma"]

    assert f1007.data.std_wide["gdp_ma"].shape[0] > 100
    assert f1007.data.std_wide["population_ma"].shape[0] > 100


queries_msa = [
    MSAFipsQuery(10180, "gdp_ma", lag=0, top=5, time_decay=1.06),
    MSAFipsQuery(
        16580,
        outcome_var="gdp_ma",
        feature_groups_with_weights={"gdp_ma": 4, "population_ma": 4},
        lag=0,
        top=5,
        time_decay=1.03,
    ),
    MSAFipsQuery(
        11020,
        feature_groups_with_weights={"gdp_ma": 4, "population_ma": 4},
        lag=0,
        top=5,
        time_decay=1.03,
    ),
    MSAFipsQuery(
        25220,
        outcome_var="gdp_ma",
        feature_groups_with_weights={"gdp_ma": 0, "population_ma": 4},
        lag=0,
        top=5,
        time_decay=1.03,
    ),
    MSAFipsQuery(39100, "gdp_ma", lag=2, top=5, time_decay=1.06),
    MSAFipsQuery(
        10580,
        outcome_var="gdp_ma",
        feature_groups_with_weights={"gdp_ma": 4, "population_ma": 4},
        lag=2,
        top=5,
        time_decay=1.03,
    ),
]


@pytest.mark.parametrize("query", queries_msa)
def test_euclidean_kins_dont_die_msa(query):
    f = query
    f.find_euclidean_kins()


# census tract level


def test_fips_query_CT_init():
    f34031124401 = CTFipsQuery(
        fips=34031124401,
        outcome_var="population",
        feature_groups_with_weights={"population": 4, "urbanicity": 4},
        lag=0,
        top=8,
    )

    assert f34031124401.outcome_var == "population"
    assert f34031124401.feature_groups == ["population", "urbanicity"]
    assert list(f34031124401.data.std_wide.keys()) == ["population", "urbanicity"]

    assert f34031124401.data.std_wide["population"].shape[0] > 100
    assert f34031124401.data.std_wide["urbanicity"].shape[0] > 100


queries_ct = [
    CTFipsQuery(45051050303, "population", lag=0, top=5, time_decay=1.06),
    CTFipsQuery(
        56033000600,
        outcome_var="population",
        feature_groups_with_weights={"population": 4, "urbanicity": 4},
        lag=0,
        top=5,
        time_decay=1.03,
    ),
    CTFipsQuery(
        6019003808,
        feature_groups_with_weights={"population": 4, "urbanicity": 4},
        lag=0,
        top=5,
        time_decay=1.03,
        ct_time_period="post_2020",
    ),
    CTFipsQuery(
        21089040100,
        outcome_var="population",
        feature_groups_with_weights={"population": 0, "urbanicity": 4},
        lag=0,
        top=5,
        time_decay=1.03,
    ),
    CTFipsQuery(
        53061051000,
        "population",
        lag=2,
        top=5,
        time_decay=1.03,
        ct_time_period="post_2020",
    ),
    CTFipsQuery(
        31047968300,
        outcome_var="population",
        feature_groups_with_weights={"population": 4, "urbanicity": 4},
        lag=2,
        top=5,
        time_decay=1.03,
        ct_time_period="post_2020",
    ),
]


@pytest.mark.parametrize("query", queries_ct)
def test_euclidean_kins_dont_die_ct(query):
    f = query
    f.find_euclidean_kins()
