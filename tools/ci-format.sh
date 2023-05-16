#!/usr/bin/env bash

set -eu

type -f clang-format || {
  echo "install clang-format"
  exit 1
}

fmt="clang-format -i {} && echo {}"
set -x
find \
  src/ -name '*.cxx' -exec sh -c "$fmt" \; \
  -or -name '*.hpp' -exec sh -c "$fmt" \;
