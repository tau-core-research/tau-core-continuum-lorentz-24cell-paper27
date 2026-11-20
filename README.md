# From Discrete Tetrahedral Clifford Pregeometry to a Four-Dimensional Lorentz-Invariant Continuum (`Paper 27` / `Sub8`, `v1.0.0`)

**Author:** Jozsef Olcsak (ORCID: `0009-0003-7311-6522`)  
**Target Journals:** *Physical Review D* / *Classical and Quantum Gravity* / *Communications in Mathematical Physics*

## Overview
Standalone bilingual (`EN` + `HU`) publication and referee verification package proving:
1. **Exact Plaquette–Unification Identity:** $\ell_*^2 = 24\pi\ell_P^2 \iff \ell_*^{-1} \equiv M_{\mathrm{Pl}}/\sqrt{3} \equiv M_* = 1.406034\times 10^{18}\;\mathrm{GeV}$.
2. **Rigid 3D $+K_4$ & $3+1\mathrm{D}$ Crystal No-Go (`Theorem 2.1`):** Excluded by LHAASO GRB 221009A ($86.8\times$), GRB vacuum birefringence ($\sim 10^{15}\times$), and CPSUV 1-loop radiative percolation.
3. **4D Spin Lift $K_4 \to 2T \to \Lambda_{F_4}\cong\Lambda_{D_4}$ 24-Cell & All-Loop $\mathrm{Pin}(W(F_4))$ Shield (`Theorem 3.1`):** Exact tensor-ring equivalence $\mathrm{Inv}_{W(F_4)}((\mathbb{R}^4)^{\otimes r}) \equiv \mathrm{Inv}_{O(4)}((\mathbb{R}^4)^{\otimes r})$ for $r \le 5$, $32$ equilateral triangular Wilson plaquettes/site, exact $\mathrm{Pin}(W(F_4))$ Overlap covariance ($1.56\times 10^{-15}$), identical vanishing of all $d \le 7$ LIV to all perturbative loop orders, and $d=8$ ($n=4$) $|\delta v_g/c| = 6.72\times 10^{-56}$ at LHAASO $18\;\mathrm{TeV}$.
4. **OS Positivity, Chiral $\mathrm{Spin}(10)$ Doublers, & Hawking–Luttrell $R^2$ + CDT (`Theorem 4.1`):** Single-slice bosonic/gauge OS positivity, $39/39$ lifted doublers, positive-action Euclidean scalaron $\chi$ + $|\nu|=1$ CDT conformal cure, and $d_S(\sigma) = 4 + \ell_*^2/(2\sigma) \to 4$.
5. **Zero-Parameter Stueckelberg–Lorenz Director Closure (`Theorem 5.1`):** Locking $f_{\mathrm{dir}} \equiv M_{\mathrm{inf}} = 3.1959\times 10^{13}\;\mathrm{GeV}$ on $c_2 = c_{\mathrm{acc}}/(1 - 2 c_{\mathrm{acc}})$ yields $c_T = c_V = c_S \equiv 1$ (zero UHECR Cherenkov loss), $\alpha_2^{\mathrm{PPN}} \equiv 0$, $\alpha_1^{\mathrm{PPN}} = -6.89\times 10^{-10}$, and $K_{\alpha\alpha}^{\mathrm{nonpert}} > 0$ from Earth to X-COP galaxy clusters.

## Deterministic Reproduction
```bash
python3 scripts/verify_paper27_continuum_lorentz_closure.py
python3 scripts/generate_paper27_figures.py
```
