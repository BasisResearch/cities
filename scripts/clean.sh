#!/bin/bash
set -euxo pipefail

isort --profile black cities/ tests/
black cities/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./cities ./tests

nbqa black docs/guides/
nbqa autoflake --remove-all-unused-imports --recursive --in-place docs/guides/ 
nbqa isort --profile black docs/guides/
