#!/usr/bin/env bash
set -euo pipefail

# Stage the canonical notebooks before watching for changes
mkdir -p notebooks

# Run the Sphinx autobuild server to watch for changes in the docs and notebooks
uv run sphinx-autobuild . _build \
  --watch ../notebooks \
  --pre-build "cp ../notebooks/*.ipynb notebooks/" \
  --re-ignore '_build/.*'