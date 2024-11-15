#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports cities/

isort --profile="black" --check --diff cities/ tests/
black --check cities/ tests/
flake8 cities/ tests/ --ignore=E203,W503 --max-line-length=127


nbqa --nbqa-shell autoflake -v --recursive --check docs/guides/
nbqa --nbqa-shell isort --profile="black" --check  docs/guides/
black --check docs/guides/
