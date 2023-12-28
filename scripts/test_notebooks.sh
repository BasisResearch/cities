#!/bin/bash

INCLUDED_NOTEBOOKS="docs/guides/"

CI=1 pytest -v --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS
