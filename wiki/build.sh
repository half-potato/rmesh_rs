#!/usr/bin/env bash
# Compile the wiki to a single PDF with tectonic.
set -euo pipefail
cd "$(dirname "$0")"
tectonic rmeshvk-wiki.tex
