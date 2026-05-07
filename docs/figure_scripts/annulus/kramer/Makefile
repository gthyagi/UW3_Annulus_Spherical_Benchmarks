PYTHON ?= /Users/tgol0006/uw_folder/uw3_git_gthyagi_latest/underworld3/.pixi/envs/amr-dev/bin/python
LATEXMK ?= latexmk

TEX_FILES := \
	figure_1_kramer_annulus_rho_ana.tex

.PHONY: all figures pdf clean

all: figures pdf clean

figures:
	$(PYTHON) run_figure_1_density_cases.py

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode figure_1_kramer_annulus_rho_ana.tex

clean:
	$(LATEXMK) -c $(TEX_FILES)
	rm -f *.bbl *.blg *.out *.synctex.gz
	rm -rf __pycache__
