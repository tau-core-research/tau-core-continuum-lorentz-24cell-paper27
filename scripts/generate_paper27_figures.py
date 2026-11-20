#!/usr/bin/env python3
"""
GENERATE PUBLICATION-GRADE VECTOR FIGURES FOR PAPER 27 / SUB8:
  fig1_24cell_WF4_isotropy_and_GRB_bounds.pdf
  fig2_overlap_doublers_and_spectral_dimension.pdf
  fig3_stueckelberg_lorenz_PPN_and_SL_stability.pdf
"""
import itertools
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.3,
    "axes.labelsize": 9.8,
    "axes.titlesize": 10.0,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.6,
    "axes.linewidth": 0.9,
})

M_PL = 2.435323e18
E_PL = M_PL * np.sqrt(8.0 * np.pi)
M_STAR = M_PL / np.sqrt(3.0)
M_INF = 3.1959e13

# =============================================================================
# FIGURE 1: (a) Tensor-Ring Invariants dim Inv_G((R^4)^{otimes r}) vs (b) Multi-Messenger LIV
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.45), gridspec_kw={"wspace": 0.27})

ranks = np.arange(1, 7)
dim_O4  = np.array([0, 1, 0, 3, 0, 15])
dim_WF4 = np.array([0, 1, 0, 3, 0, 16])
dim_B4  = np.array([0, 1, 0, 4, 0, 31])

S4_3d = []
verts_K4_3 = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=float) / np.sqrt(3.0)
for perm in itertools.permutations(range(4)):
    Pverts = verts_K4_3[list(perm)]
    M3 = Pverts.T @ np.linalg.pinv(verts_K4_3.T)
    if np.allclose(M3.T @ M3, np.eye(3), atol=1e-9):
        for s0 in [-1.0, 1.0]:
            M4 = np.zeros((4, 4))
            M4[0, 0] = s0
            M4[1:, 1:] = M3
            S4_3d.append(M4)
assert len(S4_3d) == 48
tr_TdZ2 = np.array([np.trace(M) for M in S4_3d])
dim_3DK4 = np.array([int(round(np.mean(tr_TdZ2**r))) for r in ranks])

w_bar = 0.20
ax1.bar(ranks - 1.5*w_bar, dim_3DK4, width=w_bar, color="#dc2626", alpha=0.85, label=r"3D $+K_4\times\mathbb{Z}_2^{(t)}$ ($|G|=48$)")
ax1.bar(ranks - 0.5*w_bar, dim_B4,   width=w_bar, color="#d97706", alpha=0.88, label=r"Hypercubic $W(B_4)$ ($|G|=384$)")
ax1.bar(ranks + 0.5*w_bar, dim_WF4,  width=w_bar, color="#1d4ed8", alpha=0.92, label=r"24-Cell $W(F_4)$ ($|G|=1152$)")
ax1.bar(ranks + 1.5*w_bar, dim_O4,   width=w_bar, color="#059669", alpha=0.90, label=r"Continuous $O(4)$")

ax1.set_yscale("symlog", linthresh=1.0)
ax1.set_ylim(0, 320)
ax1.set_xticks(ranks)
ax1.set_xticklabels([
    "$r=1$\n($d=3$)",
    "$r=2$\n($d=4$)",
    "$r=3$\n($d=5$)",
    "$r=4$\n($k_F,d=6$)",
    "$r=5$\n($d=7$)",
    "$r=6$\n($d=8$)"
], fontsize=8.0)
ax1.set_ylabel(r"Invariant Tensor Dimension $\dim\,\mathrm{Inv}_G((\mathbb{R}^4)^{\otimes r})$")
ax1.set_title("(a) Exact $O(4)$ Tensor-Ring Equivalence up to $r=5$")
ax1.axvspan(0.55, 5.45, color="#ecfdf5", alpha=0.50, zorder=0)

# Place legend cleanly in upper left (x = 0.6..3.3, y = 25..280) and green summary box at x = 4.35, y = 85:
ax1.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=7.2)
ax1.text(4.32, 92.0, r"$W(F_4) \equiv O(4)$" "\n" r"for $r \leq 5$:" "\n" r"zero $d \leq 7$ LIV",
         fontsize=7.6, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#059669", lw=0.85))
ax1.annotate("3D $+K_4$:\n$r=3\\to 1$", xy=(3 - 1.5*w_bar, 1.0), xytext=(2.62, 5.2),
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.85), fontsize=7.3, color="#991b1b", ha="center")
ax1.annotate("$W(B_4)$: $4 > 3$\n($k_F \\neq 0$)", xy=(4 - 0.5*w_bar, 4.0), xytext=(4.12, 13.5),
             arrowprops=dict(arrowstyle="->", color="#b45309", lw=0.85), fontsize=7.3, color="#92400e", ha="center")
ax1.grid(axis="y", ls=":", alpha=0.45)

# Panel (b): |delta v_g(E)/c| vs Energy E [GeV]:
E_grid = np.logspace(0.0, 11.3, 400)
dv_cpsuv = np.full_like(E_grid, 1.0e-3)
dv_n1    = 1.0 * (E_grid / M_STAR)**1
dv_n2    = 1.5 * (E_grid / M_STAR)**2
dv_n4    = 2.5 * (E_grid / M_STAR)**4

ax2.loglog(E_grid, dv_cpsuv, ls=":",  lw=1.5, color="#7c2d12", label=r"3+1D CPSUV 1-loop ($d=4$, $\sim g^2/16\pi^2$)")
ax2.loglog(E_grid, dv_n1,    ls="--", lw=1.7, color="#dc2626", label=r"Rigid 3D $+K_4$ ($d=5, n=1$: $E/M_*$)")
ax2.loglog(E_grid, dv_n2,    ls="-.", lw=1.6, color="#d97706", label=r"Hypercubic $W(B_4)$ ($d=6, n=2$: $\frac{3}{2}(E/M_*)^2$)")
ax2.loglog(E_grid, dv_n4,    ls="-",  lw=2.3, color="#1d4ed8", label=r"24-Cell $W(F_4)$ ($d=8, n=4$: $\frac{5}{2}(E/M_*)^4$)")

ax2.scatter([1.8e4], [1.8e4 / (10.0 * E_PL)], marker="v", s=52, color="#111827", zorder=5)
ax2.annotate("LHAASO GRB 221009A\n($18\\;\\mathrm{TeV}$ upper limit)", xy=(1.8e4, 1.47e-16), xytext=(2.8e2, 2.0e-25),
             arrowprops=dict(arrowstyle="->", color="#111827", lw=0.85), fontsize=7.4, ha="center")

dv_lhaaso_n4 = 2.5 * (1.8e4 / M_STAR)**4
ax2.scatter([1.8e4], [dv_lhaaso_n4], marker="o", s=46, color="#1d4ed8", zorder=6)
# Place annotation ABOVE-LEFT of the blue line at 18 TeV so it is completely clear of the lower-right legend!
ax2.annotate(r"$W(F_4)$ @ $18\;\mathrm{TeV}$:" "\n" r"$6.72\times 10^{-56}$" "\n" r"($\Delta t = 6.0\times 10^{-39}\;\mathrm{s}$)",
             xy=(1.8e4, dv_lhaaso_n4), xytext=(1.4e2, 4.0e-44),
             arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=0.85), fontsize=7.4, color="#1e3a8a", ha="center",
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#93c5fd", lw=0.75))

ax2.scatter([1.0e11], [1.0e-22], marker="v", s=52, color="#059669", zorder=5)
# Place Auger label cleanly between the n=2 and n=4 curves around E=10^8 GeV, y=10^-29!
ax2.annotate("Auger UHECR\n($10^{20}\\;\\mathrm{eV}$, $\\lesssim 10^{-22}$)", xy=(1.0e11, 1.0e-22), xytext=(6.0e7, 2.0e-33),
             arrowprops=dict(arrowstyle="->", color="#059669", lw=0.85), fontsize=7.4, color="#065f46", ha="center")
ax2.scatter([1.0e11], [2.5*(1.0e11/M_STAR)**4], marker="s", s=42, color="#1d4ed8", zorder=6)

ax2.set_xlim(1.0, 3.0e11)
ax2.set_ylim(1.0e-76, 1.0e2)
ax2.set_xlabel(r"Particle / Photon Energy $E$ $[\mathrm{GeV}]$")
ax2.set_ylabel(r"Fractional Group-Velocity Shift $|\delta v_g(E)/c|$")
ax2.set_title(r"(b) Multi-Messenger Dispersion ($M_* = 1.406\times 10^{18}\;\mathrm{GeV}$)")
ax2.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.1)
ax2.grid(True, which="major", ls=":", alpha=0.45)

fig.savefig(os.path.join(FIG_DIR, "fig1_24cell_WF4_isotropy_and_GRB_bounds.pdf"), bbox_inches="tight")
plt.close(fig)

# =============================================================================
# FIGURE 2: (a) 24-Cell BZ Doubler Lifting vs (b) Spectral Dimension d_S(sigma) -> 4
# =============================================================================
verts_D4 = []
for i in range(4):
    for j in range(i+1, 4):
        for si in [-1.0, 1.0]:
            for sj in [-1.0, 1.0]:
                v = np.zeros(4); v[i] = si/np.sqrt(2.0); v[j] = sj/np.sqrt(2.0)
                verts_D4.append(v)
verts_D4 = np.ascontiguousarray(verts_D4, dtype=np.float64)

def Chat_D4(K):
    K2 = np.atleast_2d(K)
    dots = np.einsum("mi,ni->mn", K2, verts_D4)
    res = (1.0 / 3.0) * np.sum(1.0 - np.cos(dots), axis=1)
    return float(res[0]) if K.ndim == 1 else res

def S_dirac_norm(K):
    K2 = np.atleast_2d(K)
    dots = np.einsum("mi,ni->mn", K2, verts_D4)
    S = (1.0 / 6.0) * np.einsum("mn,nj->mj", np.sin(dots), verts_D4)
    return np.linalg.norm(S, axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.45), gridspec_kw={"wspace": 0.27})

# Exact Brillouin-zone path passing through true BZ zeros at Chat = 0, 8, 28/3, 32/3, 0!
s_seg = np.linspace(0.0, 1.0, 220)
step_bz = np.pi / np.sqrt(2.0)
k_Gamma = step_bz * np.array([0.0, 0.0, 0.0, 0.0])
k_D1    = step_bz * np.array([0.0, 0.0, 0.0, 2.0])   # Exact doubler: Chat = 8.0, |S_Dirac| = 0!
k_D2    = step_bz * np.array([0.0, 1.0, 1.0, 2.0])   # Exact doubler: Chat = 28/3, |S_Dirac| = 0!
k_D3    = step_bz * np.array([0.0, 0.0, 2.0, 2.0])   # Exact doubler: Chat = 32/3, |S_Dirac| = 0!

path_pts = np.vstack([
    (1 - s_seg)[:, None]*k_Gamma + s_seg[:, None]*k_D1,
    (1 - s_seg)[:, None]*k_D1    + s_seg[:, None]*k_D2,
    (1 - s_seg)[:, None]*k_D2    + s_seg[:, None]*k_D3,
    (1 - s_seg)[:, None]*k_D3    + s_seg[:, None]*k_Gamma,
])
x_path = np.linspace(0.0, 4.0, len(path_pts))
s_naive = S_dirac_norm(path_pts)
c_wilson = Chat_D4(path_pts)

W_m = 0.5 * c_wilson - 2.0
ov_mag = np.sqrt(np.maximum(0.0, 2.0 * (1.0 + W_m / np.sqrt(s_naive**2 + W_m**2))))

ax1.plot(x_path, s_naive, ls="--", lw=1.8, color="#dc2626", label=r"Naive 24-cell $|S_{\mathrm{Dirac}}(k)|\ell_*$ (39 doublers)")
ax1.plot(x_path, 0.5*c_wilson, ls=":", lw=1.6, color="#059669", label=r"Wilson scalar $\frac{1}{2}\widehat{C}_{D_4}(k)$ ($\geq 4$ at doublers)")
ax1.plot(x_path, ov_mag, ls="-", lw=2.3, color="#1d4ed8", label=r"Overlap $|D_{\mathrm{ov}}(k)|\ell_*$ ($r_W=1, m_0=2$)")

for xv in [1.0, 2.0, 3.0]:
    ax1.axvline(xv, color="#9ca3af", lw=0.8, ls="-.")
ax1.scatter([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], color="#dc2626", s=34, zorder=5)
ax1.scatter([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], color="#1d4ed8", s=34, zorder=5)

ax1.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax1.set_xticklabels([
    r"$\Gamma$" + "\n" + r"$(0)$",
    r"$D_8$" + "\n" + r"$(\widehat{C}=8)$",
    r"$D_{28/3}$" + "\n" + r"$(\widehat{C}=28/3)$",
    r"$D_{32/3}$" + "\n" + r"$(\widehat{C}=32/3)$",
    r"$\Gamma$" + "\n" + r"$(0)$"
], fontsize=8.0)
ax1.set_ylim(-0.22, 6.85)
ax1.set_ylabel(r"Dimensionless Operator Norm (Lattice Units $\ell_* = 1$)")
ax1.set_title(r"(a) Chiral $\mathbf{16}_F$ Mode vs. Lifted 24-Cell Doublers")
ax1.annotate("Physical chiral $\\mathbf{16}_F$\n($|D_{\\mathrm{ov}}| \\to |k|\\ell_*$)",
             xy=(0.04, 0.05), xytext=(0.56, 2.95),
             arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=0.85), fontsize=7.5, color="#1e3a8a", ha="center",
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#93c5fd", lw=0.75))
ax1.annotate("All 39 doublers lifted\nto $|D_{\\mathrm{ov}}| = 2 M_*$",
             xy=(2.0, 2.0), xytext=(2.0, 3.05),
             arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=0.85), fontsize=7.5, color="#1e3a8a", ha="center",
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#93c5fd", lw=0.75))
ax1.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.3)
ax1.grid(axis="y", ls=":", alpha=0.45)

# Panel (b): Infinite-lattice spectral dimension d_S(sigma) on Lambda_{D_4}:
t_gl, w_gl = np.polynomial.legendre.leggauss(24)
alpha_sh = 3.2
pts_gl = (np.pi / np.sqrt(2.0)) * np.sinh(alpha_sh * t_gl) / np.sinh(alpha_sh)
wts_gl = w_gl * np.cosh(alpha_sh * t_gl)
G0, G1, G2, G3 = np.meshgrid(pts_gl, pts_gl, pts_gl, pts_gl, indexing='ij')
W0, W1, W2, W3 = np.meshgrid(wts_gl, wts_gl, wts_gl, wts_gl, indexing='ij')
Q_gl = np.ascontiguousarray(np.stack([G0.ravel(), G1.ravel(), G2.ravel(), G3.ravel()], axis=1), dtype=np.float64)
W_gl = (W0 * W1 * W2 * W3).ravel()
C_gl = Chat_D4(Q_gl)

sig_grid = np.geomspace(0.04, 200.0, 90)
dS_num = []
for sg in sig_grid:
    eb = np.exp(-sg * C_gl)
    dS_num.append(float(2.0 * sg * np.sum(W_gl * C_gl * eb) / np.sum(W_gl * eb)))
dS_num = np.array(dS_num)
sig_ir = np.geomspace(3.0, 200.0, 80)
dS_asymp = 4.0 + 1.0 / (2.0 * sig_ir)

ax2.semilogx(sig_grid, dS_num, lw=2.3, color="#1d4ed8", label=r"Exact 24-cell $\Lambda_{D_4}$ heat kernel $d_S(\sigma)$")
ax2.semilogx(sig_ir, dS_asymp, lw=1.7, ls="--", color="#dc2626", label=r"Analytic IR expansion $4 + \ell_*^2/(2\sigma)$")
ax2.axhline(4.0, color="#059669", lw=1.3, ls=":", label=r"4D Continuum Limit ($d_S^{\mathrm{IR}} = 4.0000$)")
ax2.scatter([25.0, 100.0], [4.0203, 4.0050], color="#111827", s=36, zorder=5)
ax2.annotate(r"$\sigma = 25\ell_*^2$: $d_S = 4.0203$" "\n" r"$\sigma = 100\ell_*^2$: $d_S = 4.0050$",
             xy=(25.0, 4.0203), xytext=(4.2, 2.35),
             arrowprops=dict(arrowstyle="->", color="#111827", lw=0.85), fontsize=7.6,
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#9ca3af", lw=0.8))
ax2.text(0.058, 3.05, "Discrete UV\nplaquette regime\n($\\sigma \\ll \\ell_*^2 \\Rightarrow d_S \\to 0$)",
         fontsize=7.5, color="#374151", bbox=dict(boxstyle="round,pad=0.25", fc="#f3f4f6", ec="#d1d5db", lw=0.7))
ax2.set_xlim(0.04, 200.0)
ax2.set_ylim(0.0, 4.75)
ax2.set_xlabel(r"Diffusion Scale $\sigma / \ell_*^2$")
ax2.set_ylabel(r"Spectral Dimension $d_S(\sigma) = -2\frac{\mathrm{d}\ln P(\sigma)}{\mathrm{d}\ln\sigma}$")
ax2.set_title(r"(b) Infinite-Lattice $\Lambda_{D_4}$ Spectral Dimension Flow")
ax2.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.4)
ax2.grid(True, ls=":", alpha=0.45)

fig.savefig(os.path.join(FIG_DIR, "fig2_overlap_doublers_and_spectral_dimension.pdf"), bbox_inches="tight")
plt.close(fig)

# =============================================================================
# FIGURE 3: (a) Stueckelberg-Lorenz f_dir Window vs (b) Angular Sturm-Liouville Positivity
# =============================================================================
u_K4 = np.sqrt(3.0) / 24.0
alpha_OCSR = 5.0 * u_K4
m_bg_roots = [r.real for r in np.roots([1.0, 0.0, 4.0, -np.sqrt(5.0)]) if abs(r.imag) < 1e-12 and r.real > 0]
mu_coca = (4.0 + 3.0 * (m_bg_roots[0]**2)) / (1.0 + u_K4)

def F_OD(s):
    return 1.0 + alpha_OCSR * (1.0 - np.exp(-mu_coca / s)) * np.log(1.0 + 1.0 / s)

s_scan = np.geomspace(0.05, 50.0, 20000)
ds = 1e-6 * s_scan
lam_par_vals = ((s_scan + ds)*(F_OD(s_scan + ds)**2) - (s_scan - ds)*(F_OD(s_scan - ds)**2)) / (2.0 * ds)
min_lam_par = float(np.min(lam_par_vals))
s_worst = float(s_scan[np.argmin(lam_par_vals)])
lam_perp_worst = float(F_OD(s_worst)**2)

a0_KH = 0.9015e-10
G_SI  = 6.67430e-11
c_SI  = 2.99792458e8
M_sun_kg = 1.98847e30

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.45), gridspec_kw={"wspace": 0.28})

f_grid = np.logspace(10.0, 18.0, 500)
ca_grid = (f_grid / M_PL)**2
alpha1_abs = 4.0 * ca_grid

eps_sparc = np.sqrt(G_SI * (1.0e11 * M_sun_kg) * s_worst * a0_KH) / (c_SI**2)
eps_xcop  = np.sqrt(G_SI * (3.0e13 * M_sun_kg) * s_worst * a0_KH) / (c_SI**2)
eps_extr  = np.sqrt(G_SI * (2.0e14 * M_sun_kg) * s_worst * a0_KH) / (c_SI**2)

Kmin_xcop = 2.0*ca_grid/(2.0 - ca_grid) + 2.0*(eps_xcop**2)*(min_lam_par - 1.0)
Kmin_extr = 2.0*ca_grid/(2.0 - ca_grid) + 2.0*(eps_extr**2)*(min_lam_par - 1.0)

f_max_ppn  = np.sqrt(1.0e-5) * M_PL
f_min_xcop = np.sqrt((eps_xcop**2) * (1.0 - min_lam_par)) * M_PL
f_min_extr = np.sqrt((eps_extr**2) * (1.0 - min_lam_par)) * M_PL

ax1.loglog(f_grid, alpha1_abs, lw=2.1, color="#dc2626", label=r"PPN $|\alpha_1^{\mathrm{PPN}}| = 4(f_{\mathrm{dir}}/M_{\mathrm{Pl}})^2$ ($\alpha_2 \equiv 0$)")
ax1.loglog(f_grid, np.maximum(Kmin_xcop, 1e-23), lw=2.1, color="#1d4ed8", label=r"$\min_\theta \widehat{K}_{\alpha\alpha}^{\mathrm{nonpert}}$ (X-COP $3\times 10^{13}M_\odot$)")
ax1.loglog(f_grid, np.maximum(Kmin_extr, 1e-23), lw=1.7, ls="--", color="#7c3aed", label=r"$\min_\theta \widehat{K}_{\alpha\alpha}^{\mathrm{nonpert}}$ (Extreme $2\times 10^{14}M_\odot$)")

ax1.axhline(4.0e-5, color="#dc2626", lw=1.1, ls=":")
ax1.axvspan(1.0e10, f_min_xcop, color="#fee2e2", alpha=0.55, zorder=0)
ax1.axvspan(f_max_ppn, 1.0e18, color="#fef3c7", alpha=0.65, zorder=0)
ax1.axvspan(f_min_extr, f_max_ppn, color="#ecfdf5", alpha=0.55, zorder=0)
ax1.axvline(M_INF, color="#059669", lw=2.2, ls="-")
ax1.scatter([M_INF], [4.0*(M_INF/M_PL)**2], color="#059669", s=52, zorder=6)

ax1.text(8.5e10, 8.0e-3, "Excluded:\nInhomogeneous\nSL Ghost\n($K_{\\alpha\\alpha}<0$)", fontsize=7.4, color="#991b1b", ha="center")
ax1.text(1.5e17, 1.0e-9, "Excluded:\nPPN $|\\alpha_1|$\n$> 4\\times 10^{-5}$", fontsize=7.4, color="#92400e", ha="center")
# Place green box ABOVE the curves in the clean upper-center green band (x=4e14, y=2e-2), zero overlap with lower-left legend!
ax1.annotate(r"Zero-parameter lock" "\n" r"$f_{\mathrm{dir}} \equiv M_{\mathrm{inf}} = 3.20\times 10^{13}\;\mathrm{GeV}$" "\n" r"($|\alpha_1|=6.89\times 10^{-10},\;\alpha_2\equiv 0$)",
             xy=(M_INF, 6.89e-10), xytext=(3.8e14, 2.5e-2),
             arrowprops=dict(arrowstyle="->", color="#059669", lw=1.0), fontsize=7.3, color="#065f46", ha="center",
             bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#059669", lw=0.85))

ax1.set_xlim(1.0e10, 1.0e18)
ax1.set_ylim(1.0e-19, 1.0e1)
ax1.set_xlabel(r"Director Bumblebee Scale $f_{\mathrm{dir}}$ $[\mathrm{GeV}]$")
ax1.set_ylabel(r"Dimensionless Magnitude ($|\alpha_1^{\mathrm{PPN}}|$, $\widehat{K}_{\alpha\alpha}^{\mathrm{nonpert}}$)")
ax1.set_title("(a) Dual PPN & Sturm\u2013Liouville Window ($c_T=c_V=c_S=1$)")
ax1.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.0)
ax1.grid(True, which="major", ls=":", alpha=0.45)

# Panel (b): Angular profile K_{aa}^{nonpert}(theta) / (M_Pl^2 a k^2) on s_{N,b} = 2.961 shell at f_dir = M_inf:
theta_deg = np.linspace(0.0, 90.0, 200)
th_rad = np.deg2rad(theta_deg)
mu_th = min_lam_par * (np.cos(th_rad)**2) + lam_perp_worst * (np.sin(th_rad)**2)
ca_lock = (M_INF / M_PL)**2
K0_base = 2.0 * ca_lock / (2.0 - ca_lock)

curves = [
    ("Earth / Sun / SPARC Disk ($M_b \\leq 10^{11} M_\\odot$)", eps_sparc, "#059669", "-",  2.1),
    ("X-COP Cluster Core ($M_b = 3\\times 10^{13} M_\\odot$)",  eps_xcop,  "#1d4ed8", "--", 2.1),
    ("Extreme Baryonic Limit ($M_b = 2\\times 10^{14} M_\\odot$)", eps_extr, "#7c3aed", "-.", 2.0),
]
for lbl, eps_v, col, ls_v, lw_v in curves:
    K_prof = (K0_base + 2.0 * (eps_v**2) * (np.cos(th_rad)**2) * (mu_th - 1.0)) * 1.0e10
    ax2.plot(theta_deg, K_prof, color=col, ls=ls_v, lw=lw_v, label=lbl)

ax2.axhline(0.0, color="#dc2626", lw=1.2, ls=":", label="Ghost boundary ($K_{\\alpha\\alpha}^{\\mathrm{nonpert}} = 0$)")
ax2.fill_between(theta_deg, -0.5, 0.0, color="#fee2e2", alpha=0.55)
ax2.scatter([0.0, 0.0, 0.0], [1.722, 1.567, 0.690], color=["#059669", "#1d4ed8", "#7c3aed"], s=36, zorder=5)
ax2.annotate(r"Worst-case longitudinal dip ($\theta=0^\circ$):" "\n"
             r"X-COP: $+1.57\times 10^{-10} > 0$" "\n"
             r"Extreme $2\times 10^{14}M_\odot$: $+6.90\times 10^{-11} > 0$",
             xy=(0.0, 0.690), xytext=(52.0, 0.58),
             arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=0.9), fontsize=7.4, ha="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9ca3af", lw=0.8))

ax2.set_xlim(0.0, 90.0)
ax2.set_ylim(-0.25, 3.25)
ax2.set_xlabel(r"Wavevector Angle $\theta = \arccos(\hat{\mathbf{k}}\cdot\hat{\mathbf{g}}_N)$ $[\mathrm{deg}]$")
ax2.set_ylabel(r"Scaled Kinetic Eigenvalue $10^{10}\,\frac{K_{\alpha\alpha}^{\mathrm{nonpert}}}{M_{\mathrm{Pl}}^2 a k^2}$")
ax2.set_title("(b) Angular Sturm\u2013Liouville Positivity on $s_{N,b}=2.961$ Shell")
ax2.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=7.2)
ax2.grid(True, ls=":", alpha=0.45)

fig.savefig(os.path.join(FIG_DIR, "fig3_stueckelberg_lorenz_PPN_and_SL_stability.pdf"), bbox_inches="tight")
plt.close(fig)
print("Generated all 3 vector PDFs in", FIG_DIR)
