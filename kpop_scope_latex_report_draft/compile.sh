#!/usr/bin/env bash
set -e
xelatex -interaction=nonstopmode main.tex
# If bibtex is available, run bibtex; otherwise Overleaf/TeX Live usually handles it.
if command -v bibtex >/dev/null 2>&1; then
  bibtex main
elif [ -x /usr/bin/bibtex.original ]; then
  /usr/bin/bibtex.original main
fi
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
