#!/bin/bash
set -euxo pipefail

# isort suspended till the CI-vs-local issue is resolved
# isort cities/ tests/

isort --profile="black" ./cities ./tests
black cities/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./cities ./tests

nbqa --nbqa-shell autoflake --remove-all-unused-imports --recursive --in-place docs/guides/ 
nbqa --nbqa-shell isort --profile="black" docs/guides/
black docs/guides/

