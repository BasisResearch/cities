#!/bin/bash
set -euxo pipefail

isort --profile="black" cities/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./cities ./tests
nbqa --nbqa-shell isort --profile="black" docs/guides/ 
nbqa --nbqa-shell autoflake  --nbqa-shell --remove-all-unused-imports --recursive --in-place docs/guides/   
black ./cities ./tests docs/guides/ 
