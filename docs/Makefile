LATEXMK=latexmk

all: analysis_note.pdf

analysis_note.pdf: analysis_note.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error analysis_note.tex

clean:
	$(LATEXMK) -C

.PHONY: all clean
