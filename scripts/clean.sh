#!/bin/bash
set -euxo pipefail

isort cities/ tests/
#black cities/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./cities ./tests

nbqa autoflake --remove-all-unused-imports --recursive --in-place docs/guides/ 
nbqa isort docs/guides/
#nbqa black docs/guides/

