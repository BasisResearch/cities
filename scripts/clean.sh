#!/bin/bash
set -euxo pipefail

# isort suspended as conflicting with black
# nbqa isort docs/guides/


# this sometimes conflicts with black but does some
# preliminary import sorting
# and is then overriden by black
isort cities/ tests/

black ./cities/ ./tests/ ./docs/guides/

black docs/guides/

autoflake --remove-all-unused-imports --in-place --recursive ./cities ./tests

nbqa autoflake --nbqa-shell --remove-all-unused-imports --recursive --in-place docs/guides/ 

#nbqa black docs/guides/

