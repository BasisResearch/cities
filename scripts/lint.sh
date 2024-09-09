#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports cities/ tests/
isort --check --profile="black" --diff cities/ tests/
black --check cities/ tests/ docs/guides/
flake8 cities/ tests/
nbqa --nbqa-shell autoflake --nbqa-shell --recursive --check docs/guides/
