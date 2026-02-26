# hyperbolicitydiagnostic.py
#
# Hyperbolicity diagnostic for Einstein-scalar-Gauss-Bonnet (ESGB) gravity
# following Hegade, Ripley & Yunes (arXiv:2211.08477).
#
# Computes gauge-invariant diagnostics for the loss of hyperbolicity
# in spherical symmetry.  The principal symbol for the scalar degree
# of freedom is analyzed in a local null frame (paper Sec. II C).
#
# Key outputs:
#   R_areal(r,t) — areal radius
#   M_ms(r,t)    — Misner-Sharp mass
#   det_P(r,t)   — determinant of the principal symbol [Eq. 34]
#                   det P < 0  ↔  hyperbolic
#                   det P ≥ 0  ↔  elliptic  (naked elliptic region / NER)
#   R_kk(r,t)    — outgoing null Ricci projection  R_μν k^μ k^ν
#   R_ll(r,t)    — ingoing  null Ricci projection  R_μν l^μ l^ν
#   ratio_41(r,t)— diagnostic ratio from Eq. (41); >1 ⇒ NER likely
#   ratio_36(r,t)— diagnostic ratio from Eq. (36); >1 ⇒ NER (sufficient)
#   theta_plus   — outgoing null expansion  (θ₊ = 0 locates the AH)

import numpy as np
import matplotlib.pyplot as plt

from core.grid import *
from bssn.tensoralgebra import *
from bssn.bssnvars import BSSNVars
from bssn.ModifiedGravity import GBVars, get_gb_core, get_esgb_br_terms

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Small floor to avoid division by zero
_EPS = 1e-30


# ──────────────────────────────────────────────────────────────────────────────
#  Main diagnostic function
# ──────────────────────────────────────────────────────────────────────────────

def get_hyperbolicity_diagnostic(states_over_time, t, grid, background,
                                 matter, params):
    r"""
    Compute hyperbolicity diagnostics from the full evolution data.

    Following Hegade, Ripley & Yunes (arXiv:2211.08477), we compute
    gauge-invariant diagnostics for the character of the ESGB equations
    in spherical symmetry.

    **Where** the theory breaks down is diagnosed by :math:`\det P`
    (Eq. 34).  When :math:`\det P \ge 0` a naked elliptic region (NER)
    has formed and the initial-value problem is ill-posed.

    **Why** it breaks down is diagnosed by the product of the null
    convergence scalars :math:`R_{kk} R_{ll}` vs the threshold from
    Eq. (41).  Loss of hyperbolicity is driven by strong geodesic
    focusing, *not* by violation of the null convergence condition.

    Parameters
    ----------
    states_over_time : ndarray, shape ``(num_times, NUM_VARS * N)``
        Full state vector at each output time.
    t : ndarray, shape ``(num_times,)``
        Time array.
    grid : Grid
        Spatial grid object.
    background : FlatSphericalBackground
        Background metric object.
    matter : ScalarMatter
        Matter class instance (provides ``V``, ``dVdu``, ``emtensor``).
    params : tuple
        ``(lambda_GB, a, b, chi0, coupling)`` — modified-gravity
        parameters.  ``lambda_GB`` = :math:`\ell^2`.

    Returns
    -------
    dict  (profiles have shape ``(num_times, N)``)
        ``t``          — time array
        ``r``          — coordinate radial grid
        ``R_areal``    — areal radius
        ``M_ms``       — Misner-Sharp mass
        ``sigma``      — :math:`\sigma=(D R)^2-1=-2M/R`  [Eq. 18]
        ``mu``         — :math:`\mu=R-8\ell^2 (D^c R)(D_c f)`  [Eq. 19]
        ``pi1``        — :math:`\pi_1=96\,\sigma\,(f')^2/(R^2\mu)` [Eq. 20]
        ``R_kk``       — outgoing null Ricci projection
        ``R_ll``       — ingoing  null Ricci projection
        ``det_P``      — determinant of principal symbol [Eq. 34]
        ``ratio_41``   — Eq. (41) ratio; ``>1`` suggests NER
        ``ratio_36``   — Eq. (36) ratio; ``>1`` sufficient for NER
        ``theta_plus`` — outgoing null expansion (AH at θ₊ = 0)
    """
    lambda_GB, a_mg, b_mg, chi0, coupling = params
    ell_sq = lambda_GB                       # ℓ²

    r_coord = grid.r
    N = grid.num_points
    num_times = len(t) if np.ndim(t) > 0 else 1

    # allocate output
    out_keys = ["R_areal", "M_ms", "sigma", "mu", "pi1",
                "R_kk", "R_ll", "det_P",
                "ratio_41", "ratio_36", "theta_plus"]
    out = {k: np.zeros((num_times, N)) for k in out_keys}

    for i in range(num_times):
        state = states_over_time if num_times == 1 else states_over_time[i]
        state = state.reshape(grid.NUM_VARS, -1)

        # ── BSSN variables ──────────────────────────────────────────
        bssn = BSSNVars(N)
        bssn.set_bssn_vars(state)
        matter.set_matter_vars(state, bssn, grid)

        d1 = grid.get_d1_metric_quantities(state)
        d2 = grid.get_d2_metric_quantities(state)

        # ── Modified-gravity objects ────────────────────────────────
        gb = GBVars(N)
        if lambda_GB != 0:
            get_gb_core(gb, r_coord, bssn, d1, d2, grid, background,
                        lambda_GB, chi0)
            get_esgb_br_terms(gb, r_coord, matter, bssn, d1, d2, grid,
                              background, lambda_GB, chi0, coupling)

        # ── Physical metric via tensor algebra helpers ─────────────
        em4phi = np.exp(-4.0 * bssn.phi)
        e4phi  = 1.0 / em4phi
        e2phi  = np.exp(2.0 * bssn.phi)
        K      = bssn.K

        bar_gamma_LL = get_bar_gamma_LL(r_coord, bssn.h_LL, background)
        bar_gamma_UU = get_bar_gamma_UU(r_coord, bssn.h_LL, background)
        bar_A_LL     = get_bar_A_LL(r_coord, bssn, background)

        gamma_UU = em4phi[:, np.newaxis, np.newaxis] * bar_gamma_UU
        gamma_rr_inv = gamma_UU[:, i_r, i_r]

        # ── Areal radius  R² = γ_θθ = e^{4φ} γ̄_θθ  ──────────────
        bar_gamma_tt = np.maximum(bar_gamma_LL[:, i_t, i_t], _EPS)
        sqrt_bar_gamma_tt = np.sqrt(bar_gamma_tt)
        R = e2phi * sqrt_bar_gamma_tt

        safe_R = np.maximum(np.abs(R), _EPS)

        # ∂_r R  via  ∂_r γ̄_θθ  (properly accounting for scaling_matrix)
        dphi_dr = d1.phi[:, i_r]
        d_bar_gamma_tt_dr = (background.d1_hat_gamma_LL[:, i_r, i_t, i_t]
                             + background.d1_scaling_matrix[:, i_t, i_t, i_r]
                               * bssn.h_LL[:, i_t, i_t]
                             + background.scaling_matrix[:, i_t, i_t]
                               * d1.h_LL[:, i_t, i_t, i_r])
        dR_dr = e2phi * (2.0 * dphi_dr * sqrt_bar_gamma_tt
                         + d_bar_gamma_tt_dr
                           / (2.0 * sqrt_bar_gamma_tt))

        # ── Extrinsic curvature  K_θθ ──────────────────────────────
        # K_ij = e^{4φ}(Ā_ij + (1/3) γ̄_ij K)
        bar_A_tt = bar_A_LL[:, i_t, i_t]
        K_tt = e4phi * (bar_A_tt + one_third * bar_gamma_tt * K)

        # ── Misner-Sharp mass  [Eq. 18] ────────────────────────────
        # σ = γ^rr (∂_r R)² − (K_θθ/R)² − 1
        sigma = gamma_rr_inv * dR_dr**2 - (K_tt / safe_R)**2 - 1.0
        M = -safe_R * sigma / 2.0

        # ── Coupling function derivatives from ModifiedGravity ─────
        #  gb.d1Lambdadu  = S · ℓ² · f'(u)
        #  gb.d2Lambdadduu = S · ℓ² · f''(u)
        #  where S = sigmoid smoothing.  Factor out S·ℓ² to get bare f', f''.
        u_field  = matter.u
        v_field  = matter.v
        d1_u_r   = matter.d1_u[:, i_r]

        chi = em4phi
        S   = 1.0 / (1.0 + np.exp(-100.0 * (chi - chi0)))
        S_ell = np.maximum(S * ell_sq, _EPS)
        fp  = gb.d1Lambdadu / S_ell
        fpp = gb.d2Lambdadduu / S_ell

        # ── μ = R − 8 ℓ² (D^c R)(D_c f)   [Eq. 19] ──────────────
        #  (D^c R)(D_c f) = −(n·∂R)(n·∂f) + γ^rr (∂_r R)(∂_r f)
        #  with  n·∂R = −K_θθ/R  and  n·∂f = f'·v
        n_dot_dR = -K_tt / safe_R
        DcR_Dcf  = (-n_dot_dR * (fp * v_field)
                    + gamma_rr_inv * dR_dr * (fp * d1_u_r))
        mu = R - 8.0 * ell_sq * DcR_Dcf
        safe_mu = np.where(np.abs(mu) > _EPS, mu, np.sign(mu) * _EPS)
        safe_mu = np.where(safe_mu == 0, _EPS, safe_mu)

        # ── π₁ = 96 σ (f')² / (R² μ)   [Eq. 20] ─────────────────
        pi1 = 96.0 * sigma * fp**2 / (safe_R**2 * safe_mu)

        # ── Null convergence:  R_{kk}, R_{ll}  ────────────────────
        # On-shell (leading order in ℓ): R_kk ≈ (k·∂u)²
        #   k = (n+s)/√2  (outgoing),  l = (n−s)/√2  (ingoing)
        #   s^r = 1/√γ_rr
        s_r = np.sqrt(gamma_rr_inv)
        k_dot_du = (v_field + s_r * d1_u_r) / np.sqrt(2.0)
        l_dot_du = (v_field - s_r * d1_u_r) / np.sqrt(2.0)

        R_kk = k_dot_du**2
        R_ll = l_dot_du**2

        # ── B_{kk}, B_{ll}  [Eqs. 31, 33] ────────────────────────
        # B_{kk} = R_{kk} − R/(3μ) T_{kk}
        # On-shell T_{kk} ≈ R_{kk}, so B_{kk} = R_{kk}(1 − R/(3μ))
        factor_B = 1.0 - safe_R / (3.0 * safe_mu)
        B_kk = R_kk * factor_B
        B_ll = R_ll * factor_B

        # ── T₂  (2-D trace of stress-energy)  ─────────────────────
        # For a scalar with potential:  T₂ = −2 V(u)
        T2 = -2.0 * matter.V_of_u(u_field)

        # ── λ[f] and λ[R]  (2-D d'Alembertians)  ─────────────────
        # λ[f] = f'' (D^a u)(D_a u) + f' D²u
        # Kinetic term in 2-D:  (D^a u)(D_a u) = −v² + γ^rr (∂_r u)²
        kin_2d  = -v_field**2 + gamma_rr_inv * d1_u_r**2
        DaR_Dau = -n_dot_dR * v_field + gamma_rr_inv * dR_dr * d1_u_r

        # D²u on-shell (leading order):
        # D²u = −dV/du − (2/R)(D^a R)(D_a u) − ℓ² f' G   [Eq. 25]
        # At leading order in ℓ, drop the G term.
        D2u = -matter.dVdu(u_field) - (2.0 / safe_R) * DaR_Dau
        lambda_f = fpp * kin_2d + fp * D2u

        # λ[R] from Eq. (A25):
        # λ[R] = [(4 ℓ² λ[f] − 1) σ / μ  +  R² T₂ / μ]
        lambda_R = ((4.0 * ell_sq * lambda_f - 1.0) * sigma / safe_mu
                    + safe_R**2 * T2 / safe_mu)

        # ── det P  in null frame  [Eq. 34] ────────────────────────
        # det P = π₁² ℓ⁸ R² B_ll B_kk
        #       − [1 + π₁ ℓ⁴ (λ[R] − R²/(3μ) T₂)]²
        bracket = 1.0 + pi1 * ell_sq**2 * (
            lambda_R - safe_R**2 * T2 / (3.0 * safe_mu))
        det_P = (pi1**2 * ell_sq**4 * safe_R**2 * B_ll * B_kk
                 - bracket**2)

        # ── Eq. (41) ratio  ────────────────────────────────────────
        # Sufficient condition (small ℓ):
        #   R_kk R_ll  ≥  R⁶ / (M² ℓ⁸  [128 (f')²]²)
        safe_M = np.maximum(np.abs(M), _EPS)
        fp2    = np.maximum(fp**2, _EPS)
        denom_41 = safe_M**2 * ell_sq**4 * (128.0 * fp2)**2
        ratio_41 = R_kk * R_ll * denom_41 / np.maximum(safe_R**6, _EPS)

        # ── Eq. (36) ratio  ────────────────────────────────────────
        # B_ll B_kk  ≥  R⁴ μ² / (M² ℓ⁸ [192(f')²]²)
        #             × [1 + π₁ ℓ⁴ (λ[R] − R²/(3μ) T₂)]²
        denom_36 = safe_M**2 * ell_sq**4 * (192.0 * fp2)**2
        threshold_36 = (safe_R**4 * safe_mu**2
                        / np.maximum(denom_36, _EPS) * bracket**2)
        ratio_36 = B_kk * B_ll / np.maximum(np.abs(threshold_36), _EPS)

        # ── Outgoing null expansion  θ₊  ──────────────────────────
        # θ₊ = (√2 / R)(−K_θθ/R + √γ^rr ∂_r R)
        theta_plus = (np.sqrt(2.0) / safe_R) * (-K_tt / safe_R
                                                 + s_r * dR_dr)

        # ── Store ──────────────────────────────────────────────────
        out["R_areal"][i]    = R
        out["M_ms"][i]       = M
        out["sigma"][i]      = sigma
        out["mu"][i]         = mu
        out["pi1"][i]        = pi1
        out["R_kk"][i]       = R_kk
        out["R_ll"][i]       = R_ll
        out["det_P"][i]      = det_P
        out["ratio_41"][i]   = ratio_41
        out["ratio_36"][i]   = ratio_36
        out["theta_plus"][i] = theta_plus

        for key in out_keys:
            grid.fill_inner_boundary_single_variable(out[key][i])

    out["t"] = t
    out["r"] = r_coord
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting helpers  (style matching oscillondiagnostic.py)
# ──────────────────────────────────────────────────────────────────────────────

def plot_hyperbolicity_snapshot(hyp, time_idx=-1, ax=None):
    r"""
    Reproduce the diagnostic snapshot from Fig. 4 of arXiv:2211.08477.

    Shows on a single radial axis at a chosen time:
      * :math:`-\det P`
      * characteristic-speed proxies (outgoing / ingoing null expansion)
      * :math:`r\,\alpha\, R_{\mu\nu} k^\mu k^\nu` and
        :math:`r\,\alpha\, R_{\mu\nu} l^\mu l^\nu`
      * vertical line at the NER boundary (where :math:`\det P = 0`).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    r   = hyp["r"]
    mask = r > 0

    det_P   = hyp["det_P"][time_idx]
    R_kk    = hyp["R_kk"][time_idx]
    R_ll    = hyp["R_ll"][time_idx]
    theta_p = hyp["theta_plus"][time_idx]
    t_val   = hyp["t"][time_idx] if np.ndim(hyp["t"]) > 0 else hyp["t"]

    ax.plot(r[mask], -det_P[mask],  "b-",  lw=1.4, label=r"$-\det P$")
    ax.plot(r[mask],  R_kk[mask],   "m--", lw=1.2,
            label=r"$R_{\mu\nu} k^\mu k^\nu$")
    ax.plot(r[mask],  R_ll[mask],   "r--", lw=1.2,
            label=r"$R_{\mu\nu} l^\mu l^\nu$")
    ax.plot(r[mask],  theta_p[mask], "g-", lw=1.2,
            label=r"$\theta_+$ (null expansion)")

    # NER boundary: det_P crosses zero from below
    ner_mask = det_P[mask] >= 0
    if np.any(ner_mask):
        r_ner = r[mask][ner_mask]
        for rn in [r_ner[0], r_ner[-1]]:
            ax.axvline(rn, color="k", ls=":", lw=1.0)
        ax.axvspan(r_ner[0], r_ner[-1], color="gray", alpha=0.15,
                   label="NER")

    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$r$")
    ax.set_title(f"Hyperbolicity diagnostics  (t = {t_val:.2f})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_det_P_spacetime(hyp, ax=None, vmin=None, vmax=None):
    r"""
    Color map of :math:`\det P(r, t)`.

    Blue (negative) = hyperbolic.  Red (positive) = elliptic / NER.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    r = hyp["r"]
    t = hyp["t"]
    det_P = hyp["det_P"]
    mask = r > 0

    if vmin is None:
        vmin = -np.max(np.abs(det_P[:, mask]))
    if vmax is None:
        vmax =  np.max(np.abs(det_P[:, mask]))

    pcm = ax.pcolormesh(r[mask], t, det_P[:, mask],
                        cmap="RdBu_r", vmin=vmin, vmax=vmax,
                        shading="auto")
    plt.colorbar(pcm, ax=ax, label=r"$\det P$")

    # overlay NER contour
    try:
        ax.contour(r[mask], t, det_P[:, mask], levels=[0],
                   colors="k", linewidths=1.2)
    except ValueError:
        pass

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"$\det P(r,t)$  — red = elliptic / NER")
    return ax


def plot_hyperbolicity_diagnostics(hyp, r_probe=None):
    r"""
    Four-panel summary (matches style of ``oscillondiagnostic.py``).

    [0,0]  min(det P) vs t   — negative = hyperbolic, positive = NER
    [0,1]  min(θ₊) vs t      — θ₊ = 0 marks the apparent horizon
    [1,0]  max(R_kk · R_ll) vs t  — focusing strength
    [1,1]  max(ratio_41) vs t — Eq. (41) diagnostic
    """
    t = hyp["t"]
    r = hyp["r"]
    mask = r > 0

    if r_probe is not None:
        mask = mask & (r <= r_probe)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Hyperbolicity diagnostics  (arXiv:2211.08477)", fontsize=14)

    # ── Panel 1: min(det P) ───────────────────────────────────────
    ax = axes[0, 0]
    min_det = np.min(hyp["det_P"][:, mask], axis=1)
    max_det = np.max(hyp["det_P"][:, mask], axis=1)
    ax.plot(t, min_det, "b-", lw=1.2, label=r"$\min(\det P)$")
    ax.plot(t, max_det, "r--", lw=1.0, label=r"$\max(\det P)$")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.fill_between(t, 0, max_det, where=max_det > 0,
                    color="red", alpha=0.15)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\det P$")
    ax.set_title(r"Principal-symbol determinant")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: min(θ₊) ─────────────────────────────────────────
    ax = axes[0, 1]
    min_theta = np.min(hyp["theta_plus"][:, mask], axis=1)
    ax.plot(t, min_theta, "g-", lw=1.2)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\min(\theta_+)$")
    ax.set_title(r"Outgoing null expansion  ($\theta_+ = 0$ → AH)")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: max(R_kk · R_ll) ────────────────────────────────
    ax = axes[1, 0]
    prod = hyp["R_kk"][:, mask] * hyp["R_ll"][:, mask]
    max_prod = np.max(prod, axis=1)
    pos = max_prod > 0
    y = np.full_like(max_prod, np.nan)
    y[pos] = np.log10(max_prod[pos])
    ax.plot(t, y, "m-", lw=1.2)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\log_{10}(\max\, R_{kk} R_{ll})$")
    ax.set_title(r"Geodesic focusing  $R_{kk} R_{ll}$")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: max(ratio_41) ────────────────────────────────────
    ax = axes[1, 1]
    max_r41 = np.max(hyp["ratio_41"][:, mask], axis=1)
    pos41 = max_r41 > 0
    y41 = np.full_like(max_r41, np.nan)
    y41[pos41] = np.log10(max_r41[pos41])
    ax.plot(t, y41, "k-", lw=1.2)
    ax.axhline(0, color="r", lw=0.8, ls="--", alpha=0.7,
               label="threshold (ratio = 1)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{ratio}_{41})$")
    ax.set_title(r"Eq. (41) diagnostic ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_null_convergence_profiles(hyp, times=None, ax=None):
    r"""
    Radial profiles of the null convergence scalars at selected times.

    * Solid = :math:`R_{\mu\nu} k^\mu k^\nu`  (outgoing)
    * Dashed = :math:`R_{\mu\nu} l^\mu l^\nu` (ingoing)

    NCC is violated where either of these is negative.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    r   = hyp["r"]
    t   = hyp["t"]
    mask = r > 0

    if times is None:
        times = np.linspace(t[0], t[-1], 6)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(times), vmax=max(times))

    for ti in times:
        idx = np.argmin(np.abs(t - ti))
        c   = cmap(norm(ti))
        ax.plot(r[mask], hyp["R_kk"][idx, mask], "-",
                color=c, lw=1.2, label=rf"$R_{{kk}}$, t={t[idx]:.1f}")
        ax.plot(r[mask], hyp["R_ll"][idx, mask], "--",
                color=c, lw=1.0)

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$R_{\mu\nu}\, n^\mu n^\nu$")
    ax.set_title("Null convergence  (solid = outgoing, dashed = ingoing)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def plot_misner_sharp_mass(hyp, times=None, ax=None):
    """Radial profiles of the Misner-Sharp mass at selected times."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    r   = hyp["r"]
    t   = hyp["t"]
    mask = r > 0

    if times is None:
        times = np.linspace(t[0], t[-1], 6)

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=min(times), vmax=max(times))

    for ti in times:
        idx = np.argmin(np.abs(t - ti))
        c   = cmap(norm(ti))
        ax.plot(r[mask], hyp["M_ms"][idx, mask], color=c, lw=1.2,
                label=f"t = {t[idx]:.1f}")

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$M_{\mathrm{MS}}$")
    ax.set_title("Misner-Sharp mass")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax
