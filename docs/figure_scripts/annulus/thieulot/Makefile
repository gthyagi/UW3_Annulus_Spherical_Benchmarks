PYTHON ?= /Users/tgol0006/uw_folder/uw3_git_gthyagi_latest/underworld3/.pixi/envs/amr-dev/bin/python
LATEXMK ?= latexmk

TEX_FILES := \
	figure_1_thieulot_annulus_analytical.tex \
	table_figure_5_thieulot_annulus_convergence.tex \
	thieulot_annulus_benchmark_article.tex \
	thieulot_annulus_benchmark_report.tex

.PHONY: all field-plots figures pdf clean

all: field-plots figures pdf clean

field-plots:
	$(PYTHON) run_field_plot_cases.py

figures:
	$(PYTHON) plot_figure_2_thieulot_annulus_pressure_boundary.py
	$(PYTHON) plot_figure_5_thieulot_annulus_convergence.py
	$(PYTHON) plot_p2p1_boundary_pressure_convergence.py
	$(PYTHON) plot_boundary_pressure_error_by_element_h128.py
	$(PYTHON) make_table_figure_5_thieulot_annulus_convergence.py
	$(LATEXMK) -pdf -interaction=nonstopmode table_figure_5_thieulot_annulus_convergence.tex

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode figure_1_thieulot_annulus_analytical.tex
	$(LATEXMK) -pdf -interaction=nonstopmode thieulot_annulus_benchmark_article.tex
	$(LATEXMK) -pdf -interaction=nonstopmode thieulot_annulus_benchmark_report.tex

clean:
	$(LATEXMK) -c $(TEX_FILES)
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz
	rm -rf __pycache__
