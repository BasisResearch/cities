#!/bin/bash
set -euxo pipefail

<<<<<<< HEAD
isort --profile="black" cities/ tests/
black cities/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./cities ./tests

nbqa --nbqa-shell autoflake --remove-all-unused-imports --recursive --in-place docs/guides/ docs/testing_notebooks
nbqa --nbqa-shell isort --profile="black" docs/guides/ docs/testing_notebooks
black docs/guides/ docs/testing_notebooks
=======
# isort suspended till the CI-vs-local issue is resolved
# isort cities/ tests/

black cities/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./cities ./tests

nbqa autoflake --remove-all-unused-imports --recursive --in-place docs/guides/ 
# nbqa isort docs/guides/
nbqa black docs/guides/
>>>>>>> e3a66ed4029913c0706d064001cdfede0cc6f413

