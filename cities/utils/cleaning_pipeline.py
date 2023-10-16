import sys

from cities.utils.clean_gdp import clean_gdp
from cities.utils.clean_population import clean_population
from cities.utils.clean_transport import clean_transport

# from cities.utils.cleaning_utils import find_repo_root
# sys.path.insert(0, find_repo_root())


clean_gdp()

clean_population()

clean_transport()
