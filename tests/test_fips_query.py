from cities.utils.data_grabber import DataGrabber
data = DataGrabber()

from cities.utils.fips_query import FipsQuery


def test_fips_query_init():

    f1007 = FipsQuery(fips = 1007, outcome_var= "gdp",
                    feature_groups= ["population"],
                    lag = 0, top = 8)

    assert f1007.outcome_var == "gdp"
    assert f1007.feature_groups == ["population"]
    assert f1007.weights == [4]
    assert list(f1007.data.std_wide.keys()) == ["gdp", "population"]

    assert f1007.data.std_wide['gdp'].shape[0] > 100
    assert f1007.data.std_wide['population'].shape[0] > 100 
