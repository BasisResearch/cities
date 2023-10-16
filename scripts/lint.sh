#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports cities/
isort --check --profile black --diff cities/ tests/
black --check cities/ tests/
# stop the build if there are Python syntax errors or undefined names
flake8 cities/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 cities/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics