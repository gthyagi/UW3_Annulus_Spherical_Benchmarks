PYTHON ?= /Users/tgol0006/uw_folder/uw3_git_gthyagi_latest/underworld3/.pixi/envs/amr-dev/bin/python
LATEXMK ?= latexmk

TEX_FILES := \
	figure_2_kramer_rho_ana.tex \
	table_figure_4_kramer_convergence.tex \
	table_figure_7_kramer_boundary_convergence.tex \
	kramer_spherical_benchmark_report.tex

.PHONY: all figures tables report clean

all: figures tables report clean

figures:
	$(LATEXMK) -pdf -interaction=nonstopmode figure_2_kramer_rho_ana.tex
	$(PYTHON) plot_figure_4_kramer_convergence.py
	$(PYTHON) plot_figure_7_kramer_boundary_convergence.py

tables:
	$(PYTHON) make_table_figure_4_kramer_convergence.py
	$(PYTHON) make_table_figure_7_kramer_boundary_convergence.py
	$(LATEXMK) -pdf -interaction=nonstopmode table_figure_4_kramer_convergence.tex
	$(LATEXMK) -pdf -interaction=nonstopmode table_figure_7_kramer_boundary_convergence.tex

report:
	$(LATEXMK) -pdf -interaction=nonstopmode kramer_spherical_benchmark_report.tex

clean:
	$(LATEXMK) -c $(TEX_FILES)
	rm -f *.bbl *.blg *.out *.synctex.gz
	rm -rf __pycache__
