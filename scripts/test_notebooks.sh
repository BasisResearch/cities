#!/bin/bash

INCLUDED_NOTEBOOKS="docs/guides/ " # docs/testing_notebooks/"  will revert when the pyro-ppl 1.9 bug is fixed

CI=1 pytest -v --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS
