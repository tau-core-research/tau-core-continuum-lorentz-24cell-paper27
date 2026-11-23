# Paper 27: English referee revision

Finite-Rank Invariants of the 24-Cell and Lattice Field Diagnostics

The finite-rank selection rule, anisotropic rank-six counterexample, and positive scalar site-reflection result do not establish the interacting chiral/gauge measure or any continuum gravity reconstruction.

This revision contains the corrected English preprint and a source ZIP.
It does not claim journal acceptance. Earlier bilingual or submission
packages are historical; the Hungarian version has not been revised
or included in this update.

Build the PDF from the extracted ZIP with:

```sh
cd paper27_submission_source
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The ZIP also preserves the corresponding benchmark scripts with
`requirements.txt`. Those scripts run the named benchmarks; their
output labels do not override the claim boundaries of the revised text.
For the new scalar controls, run `python3 -W error scripts/verify_d4_scalar_reflection.py` from the source ZIP root.
