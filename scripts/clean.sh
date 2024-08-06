#!/bin/bash
set -euxo pipefail

isort --profile black cities/ tests/
black cities/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./cities ./tests

nbqa black docs/guides/
nbqa autoflake --remove-all-unused-imports --recursive --in-place docs/guides/
nbqa isort -in-place docs/guides/

pg_format -c .pg_format -i etl/*.sql
