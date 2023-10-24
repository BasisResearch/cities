#!/bin/bash
set -euxo pipefail

./scripts/lint.sh
cd tests && pytest