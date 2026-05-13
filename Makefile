# Makefile for the research site.
#
# Targets:
#   make install   — create a venv and install Python deps
#   make preview   — local server with live reload (Quarto's `quarto preview`)
#   make           — full render to `_site/` (equivalent to `make render`)
#   make render    — same as `make`
#   make pdf       — render the paper-only PDF alongside the website
#   make clean     — wipe rendered output and the freeze cache
#   make publish   — push gh-pages branch via `quarto publish gh-pages`
#
# Quarto must be installed system-wide: https://quarto.org/docs/get-started/

.PHONY: default install preview render pdf clean publish

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

default: render

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: $(VENV)/bin/activate ## Create the venv and install Python deps

preview: install
	. $(VENV)/bin/activate && quarto preview

render: install
	. $(VENV)/bin/activate && quarto render

# Renders just the paper chapters as a single PDF. Requires a TeX install
# (tinytex is the easiest — Quarto bundles it via `quarto install tinytex`).
pdf: install
	. $(VENV)/bin/activate && quarto render paper --to pdf

clean:
	rm -rf _site _freeze _artifacts .quarto

# Quarto's GH Pages publish workflow: builds, switches to the gh-pages
# branch, force-pushes the rendered _site/ contents. Use this only if you
# don't want the GitHub Action — otherwise just push to main.
publish: install
	. $(VENV)/bin/activate && quarto publish gh-pages
