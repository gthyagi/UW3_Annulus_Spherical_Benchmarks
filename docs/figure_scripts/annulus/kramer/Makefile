PYTHON ?= /Users/tgol0006/uw_folder/uw3_git_gthyagi_latest/underworld3/.pixi/envs/amr-dev/bin/python
LATEXMK ?= latexmk

TEX_FILES := \
	figure_1_kramer_annulus_rho_ana.tex \
	table_figure_3_kramer_annulus_convergence.tex \
	table_boundary_pressure_convergence.tex \
	table_boundary_pressure_convergence_rates.tex \
	kramer_annulus_benchmark_article.tex \
	kramer_annulus_benchmark_report.tex

.PHONY: all figures tables pdf article report clean

all: figures tables pdf article report clean

figures:
	$(PYTHON) run_figure_1_density_cases.py
	$(PYTHON) plot_figure_3_kramer_annulus_convergence.py
	$(PYTHON) plot_boundary_pressure_convergence.py

tables:
	$(PYTHON) make_table_figure_3_kramer_annulus_convergence.py
	$(LATEXMK) -pdf -interaction=nonstopmode table_figure_3_kramer_annulus_convergence.tex
	$(PYTHON) make_table_boundary_pressure_convergence.py
	$(LATEXMK) -pdf -interaction=nonstopmode table_boundary_pressure_convergence.tex
	$(PYTHON) make_table_boundary_pressure_convergence_rates.py
	$(LATEXMK) -pdf -interaction=nonstopmode table_boundary_pressure_convergence_rates.tex

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode figure_1_kramer_annulus_rho_ana.tex

report:
	$(LATEXMK) -pdf -interaction=nonstopmode kramer_annulus_benchmark_report.tex

article:
	$(LATEXMK) -pdf -interaction=nonstopmode kramer_annulus_benchmark_article.tex

clean:
	$(LATEXMK) -c $(TEX_FILES)
	rm -f *.bbl *.blg *.out *.synctex.gz
	rm -rf __pycache__
