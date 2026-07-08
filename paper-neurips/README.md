# NeurIPS 2024 version

Same paper as `../paper/`, formatted with the official NeurIPS 2024 style
(`neurips_2024.sty`, from media.neurips.cc). Retarget to the current year by
swapping the style file when the CFP opens.

## Compile
```bash
pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex
```

## Modes (top of paper.tex)
- `\usepackage[preprint]{neurips_2024}` — named preprint (current).
- `\usepackage{neurips_2024}` — anonymous submission.
- `\usepackage[final]{neurips_2024}` — camera-ready.
The first line `\PassOptionsToPackage{numbers}{natbib}` keeps numeric citations.
