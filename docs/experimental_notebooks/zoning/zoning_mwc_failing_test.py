import contextlib
from typing import Dict, List, Optional

import pyro
import pyro.distributions as dist
import torch
from cities.modeling.simple_linear import SimpleLinear


import copy
