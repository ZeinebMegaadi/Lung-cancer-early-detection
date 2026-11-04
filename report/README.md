# report/

LaTeX source of the midterm report (revised version after midterm feedback). The tables that hold numbers and the
key numbers in the prose are **generated from the pipeline outputs**, so the report cannot drift from the analysis.

```
main.tex                         preamble, title page, \input of every section
sections/                        the prose, one file per section (00-titlepage ... 08-appendix, references)
generated/                       written by scripts/build_report.py - never edit by hand
  numbers.tex                    \newcommand macros for the key numbers (\nFeatures, \maxVIF, \pcaNinetyFive, ...)
  tab_*.tex                      the body (tabular) of every table built from reports/tables/*.csv and data/clean/*.csv
Lung_Cancer_Midterm_Report.pdf   the compiled report (copy of build/main.pdf)
build/                           pdflatex output (git-ignored)
```

The figures are read directly from `../reports/figures/` (`\graphicspath` in `main.tex`); there is no copy here.

## Build

Needs `pdflatex` on the PATH: MiKTeX (Windows; let it install missing packages on the fly) or TeX Live. All packages
used are standard: geometry, booktabs, tabularx, makecell, caption, titlesec, fancyhdr, hyperref, xspace,
newunicodechar, lmodern, microtype, amsmath, enumitem, float.

```bash
python run_all.py                  # the analysis (if the outputs are not up to date)
python scripts/build_report.py     # writes generated/, then runs pdflatex twice -> build/main.pdf
```

**VS Code (LaTeX Workshop):** open `main.tex` and pick the recipe **pdflatex x2** (Ctrl+Alt+B). It runs
`pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex` twice, so cross-references
resolve. Do not use the latexmk recipes: MiKTeX's latexmk needs Perl. The recipe lives in `.vscode/settings.json`,
which is not committed; to set it up, add:

```json
"latex-workshop.latex.tools": [{"name": "pdflatex", "command": "pdflatex",
  "args": ["-interaction=nonstopmode", "-halt-on-error", "-output-directory=%OUTDIR%", "%DOC%"]}],
"latex-workshop.latex.recipes": [{"name": "pdflatex x2", "tools": ["pdflatex", "pdflatex"]}],
"latex-workshop.latex.recipe.default": "pdflatex x2",
"latex-workshop.latex.outDir": "%DIR%/build"
```

After a build, copy `build/main.pdf` to `Lung_Cancer_Midterm_Report.pdf` to update the committed copy.

## Conventions

- Captions, labels and prose live in `sections/`; `generated/tab_*.tex` holds only the `tabular` body.
- Every table, figure and section has a `\label`; text always refers with `\ref`, never a typed number.
- Figures 12b and 12c keep those numbers (`\figsuffix{b}`), because the figure number is printed in each PNG's title.
- Hand-written tables (no CSV behind them): Tables 1, 2 (research questions, criteria), sources, data-quality
  problems, integrity audit, regional cleaning, engineered features, answers to the research questions, pipeline steps.
