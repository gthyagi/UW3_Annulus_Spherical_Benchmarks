PYTHON ?= /Users/tgol0006/uw_folder/uw3_git_gthyagi_latest/underworld3/.pixi/envs/amr-dev/bin/python
LATEXMK ?= latexmk

TEX_FILES := \
	figure_2_thieulot_analytical.tex \
	table_figures_4_5_convergence.tex \
	table_boundary_metric_convergence.tex \
	thieulot_spherical_benchmark_article.tex \
	thieulot_spherical_benchmark_report.tex

.PHONY: all figures tables article report clean

all: figures tables article report clean

figures:
	$(PYTHON) plot_figures_4_5_convergence.py
	$(PYTHON) plot_boundary_metric_convergence.py

tables:
	$(LATEXMK) -pdf -interaction=nonstopmode table_figures_4_5_convergence.tex
	$(LATEXMK) -pdf -interaction=nonstopmode table_boundary_metric_convergence.tex

report:
	$(LATEXMK) -pdf -interaction=nonstopmode thieulot_spherical_benchmark_report.tex

article:
	$(LATEXMK) -pdf -interaction=nonstopmode thieulot_spherical_benchmark_article.tex

clean:
	$(LATEXMK) -c $(TEX_FILES)
	rm -f *.bbl *.blg *.out *.synctex.gz
	rm -rf __pycache__
