#!/bin/bash
set -euxo pipefail

isort --profile="black" cities/ tests/
black cities/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./cities ./tests

nbqa --nbqa-shell autoflake --remove-all-unused-imports --recursive --in-place docs/guides/ docs/testing_notebooks
nbqa --nbqa-shell isort --profile="black" docs/guides/ docs/testing_notebooks
black docs/guides/ docs/testing_notebooks

