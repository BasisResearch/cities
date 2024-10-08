#!/bin/bash
set -euxo pipefail


path="${1:-.}"

isort --profile="black" "$path"
black "$path"
autoflake --remove-all-unused-imports --in-place --recursive "$path"

if [[ -d "$path" ]]; then
  nbqa --nbqa-shell autoflake --remove-all-unused-imports --recursive --in-place "$path"
  nbqa --nbqa-shell isort --profile="black" "$path"
  black "$path"
fi
