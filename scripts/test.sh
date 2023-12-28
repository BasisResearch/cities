#!/bin/bash
set -euxo pipefail

CI=1 cd tests && pytest