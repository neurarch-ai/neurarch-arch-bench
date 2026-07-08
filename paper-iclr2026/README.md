# ICLR 2026 version

Same paper as `../paper/`, formatted with the official ICLR 2026 template.

## Compile
```bash
pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex   # 3 passes for citations
```
Self-contained: the ICLR style files (`iclr2026_conference.sty`, `fancyhdr.sty`,
`natbib.sty`, `.bst`) are bundled, from github.com/ICLR/Master-Template (iclr2026).

## Anonymous vs named
Submission is anonymous by default. For a camera-ready or arXiv preprint that
shows the author, uncomment `\iclrfinalcopy` near `\begin{document}`.
