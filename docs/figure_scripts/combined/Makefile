LATEXMK ?= latexmk
MAGICK ?= magick

DENSITY_FIG := combined_density_distribution_figures
JPG_DPI ?= 500
JPG_QUALITY ?= 96

SOURCE_FIGURES := \
	../annulus/thieulot/figure_1_thieulot_annulus_analytical.pdf \
	../annulus/kramer/figure_1_kramer_annulus_rho_ana.pdf \
	../spherical/thieulot/figure_2_thieulot_analytical.pdf \
	../spherical/kramer/figure_2_kramer_rho_ana.pdf

.PHONY: all clean

all: $(DENSITY_FIG).pdf $(DENSITY_FIG).jpg

$(DENSITY_FIG).pdf: $(DENSITY_FIG).tex $(SOURCE_FIGURES)
	$(LATEXMK) -pdf -interaction=nonstopmode $<

$(DENSITY_FIG).jpg: $(DENSITY_FIG).pdf
	$(MAGICK) -density $(JPG_DPI) '$<[0]' -background white -alpha remove -alpha off -quality $(JPG_QUALITY) $@

clean:
	$(LATEXMK) -c $(DENSITY_FIG).tex
