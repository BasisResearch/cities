import sys
import os

from cleaning_utils import find_repo_root
sys.path.insert(0, find_repo_root())

from cleaning_utils import clean_gdp



clean_gdp()
