#!/bin/bash
set -euxo pipefail

isort --profile black cities/ tests/
black cities/ tests/
