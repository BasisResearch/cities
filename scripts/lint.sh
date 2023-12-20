#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports cities/
isort --check --profile black --diff cities/ tests/
black --check cities/ tests/
flake8 cities/ tests/ --ignore=E203,W503 --max-line-length=127

nbqa black --check docs/guides/
nbqa autoflake --recursive --check docs/guides/
nbqa isort --check docs/guides/