#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Syncs only the flat *.ipynb files directly in this folder (their paired
# .py:percent twins) — does NOT touch preprocessing/ or other/.
#
# To pair a new notebook for the first time (one-off, run manually):
#   jupytext --set-formats ipynb,py:percent notebooks/NewOne.ipynb
jupytext --sync *.ipynb