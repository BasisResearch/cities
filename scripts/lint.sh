#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports cities/
#isort --check --diff cities/ tests/
black --check cities/ tests/ docs/guides/
flake8 cities/ tests/ --ignore=E203,W503 --max-line-length=127


nbqa autoflake --nbqa-shell -v --recursive --check docs/guides/
#nbqa isort --check  docs/guides/

