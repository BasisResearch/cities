#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports cities/
<<<<<<< HEAD
isort --profile="black" --check --diff cities/ tests/
=======
#isort --check --diff cities/ tests/
>>>>>>> e3a66ed4029913c0706d064001cdfede0cc6f413
black --check cities/ tests/
flake8 cities/ tests/ --ignore=E203,W503 --max-line-length=127


<<<<<<< HEAD
nbqa --nbqa-shell autoflake -v --recursive --check docs/guides/
nbqa --nbqa-shell isort --profile="black" --check  docs/guides/
black --check docs/guides/
=======
nbqa autoflake -v --recursive --check docs/guides/
#nbqa isort --check  docs/guides/
nbqa black --check docs/guides/
>>>>>>> e3a66ed4029913c0706d064001cdfede0cc6f413
