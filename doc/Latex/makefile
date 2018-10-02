SHELL=/bin/sh
THESIS=template_tesis_maie

.SUFFIXES:
.SUFFIXES: .bib .pdf .tex
.PHONY: clean

run: $(THESIS).pdf

$(THESIS).pdf: $(THESIS).bbl $(THESIS).tex
	pdflatex $(THESIS).tex -draftmode
	pdflatex $(THESIS).tex 

$(THESIS).bbl: $(THESIS).aux
	bibtex $(THESIS).aux

$(THESIS).aux: $(THESIS).bib
	pdflatex $(THESIS).tex -draftmode
	pdflatex $(THESIS).tex -draftmode

clean:

	rm -f *.aux *.log *.toc *.backup *.out *.lof *.lot *.zip *.bbl *.blg *.nlo *~ secciones/*aux  secciones/*.bak *.bak bibliografia/*.bak

cleanall: clean

	rm -f *.pdf

open: $(THESIS).pdf

	okular $(THESIS)).pdf &

package: $(THESIS).pdf 
	(find . -name "*.tex" && find . -name "makefile" && find . -name "maiedoc.sty" && find . -name "*.bib" && find . -name "*.bst" && find ./imagenes -name "*" && find ./secciones -name "*.pdf") | zip $(THESIS).zip -@ \
