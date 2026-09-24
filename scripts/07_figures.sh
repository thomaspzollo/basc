#!/usr/bin/env bash
# Step 7 (CPU): execute the paper notebooks. Figures are written to figures/,
# LaTeX tables to figures/tables/, and the executed notebooks (with outputs) to
# figures/executed_notebooks/, leaving notebooks/ unchanged.
source "$(dirname "$0")/config.sh"

for nb in notebooks/[0-9]*.ipynb; do
  echo "[run] $nb"
  $PYTHON -m nbconvert --to notebook --execute "$nb" --output-dir "${BASC_FIGURES_DIR:-figures}/executed_notebooks"
done
