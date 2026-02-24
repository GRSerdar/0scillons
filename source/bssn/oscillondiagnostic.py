# oscillondiagnostic.py
#
# Oscillon diagnostics following Aurrekoetxea, Clough & Muia (arXiv:2304.01673)
#
# Computes:  rho(r,t), rho_c(t), rho_bar(t), delta_c(t), a(t),
#            oscillon mass M(t), volume V(t), radius R(t), compactness C(t)
#
# Usage mirrors constraintsdiagnostic.py — pass the full solution array and
# the same grid / background / matter / params objects used in the evolution.

import numpy as np
import matplotlib.pyplot as plt

from core.grid import *
from bssn.tensoralgebra import *
from bssn.bssnvars import BSSNVars
from bssn.ModifiedGravity import GBVars, get_gb_core, get_esgb_br_terms


# NumPy >= 2.0 renamed trapz -> trapezoid
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ──────────────────────────────────────────────────────────────────────────────
#  Main diagnostic function
# ──────────────────────────────────────────────────────────────────────────────

def get_oscillon_diagnostic(states_over_time, t, grid, background, matter,
                            params, surface_threshold=0.05):
    """
    Compute oscillon diagnostics from the full evolution data.

    Parameters
    ----------
    states_over_time : ndarray, shape (num_times, NUM_VARS * N)
        Full state vector at each output time.  This must be the FULL
        solution (not background-subtracted) because physical quantities
        like rho, M and C require the actual field values.
    t : ndarray, shape (num_times,)
        Time array.
    grid : Grid
        Spatial grid object.
    background : FlatSphericalBackground
        Background metric object.
    matter : ScalarMatter
        Matter class instance (provides V, dV/du, emtensor).
    params : tuple
        (lambda_GB, a, b, chi0, coupling) — modified-gravity parameters.
    surface_threshold : float
        Fraction of central density that defines the oscillon surface
        (default 5 %, matching the paper Sec. II-C).

    Returns
    -------
    dict with keys:
        t         – time array
        r         – radial grid
        rho       – energy density profiles, shape (num_times, N)
        rho_c     – central (max) energy density, shape (num_times,)
        rho_bar   – volume-averaged energy density, shape (num_times,)
        delta_c   – density contrast  rho_c / rho_bar - 1
        a         – scale factor from volume-averaged conformal factor
        M         – oscillon mass  (integrated over rho > threshold * rho_c)
        V_proper  – proper volume of the oscillon
        R         – oscillon radius  (3 V / 4 pi)^{1/3}
        C         – compactness  G M / R   (G = 1/(8 pi) in code units)
    """
    lambda_GB, a_mg, b_mg, chi0, coupling = params

    r = grid.r
    N = grid.num_points
    num_times = len(t) if np.ndim(t) > 0 else 1

    rho_all   = np.zeros((num_times, N))
    rho_c     = np.zeros(num_times)
    rho_bar   = np.zeros(num_times)
    delta_c   = np.zeros(num_times)
    scale_fac = np.zeros(num_times)
    mass      = np.zeros(num_times)
    vol       = np.zeros(num_times)
    radius    = np.zeros(num_times)
    compact   = np.zeros(num_times)

    r_phys_mask = r > grid.min_dr * 0.5

    for i in range(num_times):
        if num_times == 1:
            state = states_over_time
        else:
            state = states_over_time[i]

        state = state.reshape(grid.NUM_VARS, -1)

        # ── BSSN variables ────────────────────────────────────────────────
        bssn_vars = BSSNVars(N)
        bssn_vars.set_bssn_vars(state)
        matter.set_matter_vars(state, bssn_vars, grid)

        d1 = grid.get_d1_metric_quantities(state)
        d2 = grid.get_d2_metric_quantities(state)

        # ── Modified-gravity objects (needed for rho_GB) ──────────────────
        gb = GBVars(N)
        if lambda_GB != 0:
            get_gb_core(gb, r, bssn_vars, d1, d2, grid, background,
                        lambda_GB, chi0)
            get_esgb_br_terms(gb, r, matter, bssn_vars, d1, d2, grid,
                              background, lambda_GB, chi0, coupling)

        # ── Energy-momentum tensor ────────────────────────────────────────
        emtensor = matter.get_emtensor(r, bssn_vars, background, gb)
        rho = emtensor.rho + gb.rho_GB

        rho_all[i, :] = rho

        # ── Conformal factor quantities ───────────────────────────────────
        phi    = bssn_vars.phi
        e6phi  = np.exp(6.0 * phi)
        e4phi  = np.exp(4.0 * phi)

        # ── Volume integrals (spherical symmetry, 4 pi r^2 factor) ───────
        rp    = r[r_phys_mask]
        rho_p = rho[r_phys_mask]
        e6p   = e6phi[r_phys_mask]
        e4p   = e4phi[r_phys_mask]

        vol_element = 4.0 * np.pi * rp**2 * e6p
        total_vol   = _trapz(vol_element, rp)

        rho_bar[i] = _trapz(rho_p * vol_element, rp) / max(total_vol, 1e-30)
        rho_c[i]   = np.max(rho)
        delta_c[i] = rho_c[i] / max(rho_bar[i], 1e-30) - 1.0

        coord_vol = _trapz(4.0 * np.pi * rp**2, rp)
        avg_e4phi = _trapz(e4p * 4.0 * np.pi * rp**2, rp) / max(coord_vol, 1e-30)
        scale_fac[i] = np.sqrt(max(avg_e4phi, 1e-30))

        # ── Oscillon mass, volume, radius, compactness ────────────────────
        osc_mask = r_phys_mask & (rho > surface_threshold * rho_c[i])
        if np.any(osc_mask):
            rp_osc  = r[osc_mask]
            rho_osc = rho[osc_mask]
            e6_osc  = e6phi[osc_mask]

            dV_osc      = 4.0 * np.pi * rp_osc**2 * e6_osc
            mass[i]     = _trapz(rho_osc * dV_osc, rp_osc)
            vol[i]      = _trapz(dV_osc, rp_osc)
            radius[i]   = (3.0 * vol[i] / (4.0 * np.pi))**(1.0 / 3.0)
            if radius[i] > 0:
                compact[i] = mass[i] / (8.0 * np.pi * radius[i])

        grid.fill_inner_boundary_single_variable(rho_all[i, :])

    return {
        "t"        : t,
        "r"        : r,
        "rho"      : rho_all,
        "rho_c"    : rho_c,
        "rho_bar"  : rho_bar,
        "delta_c"  : delta_c,
        "a"        : scale_fac,
        "M"        : mass,
        "V_proper" : vol,
        "R"        : radius,
        "C"        : compact,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting helpers  (mimic figures from the paper)
# ──────────────────────────────────────────────────────────────────────────────

def plot_density_profiles_at_times(osc, times=None, ax=None):
    """
    Plot radial energy-density profiles rho(r) at selected coordinate times.
    """
    if times is None:
        times = np.linspace(osc["t"][0], osc["t"][-1], 6)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    r = osc["r"]
    t_arr = osc["t"]
    mask = r > 0

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(times), vmax=max(times))

    for ti in times:
        idx = np.argmin(np.abs(t_arr - ti))
        color = cmap(norm(ti))
        ax.plot(r[mask], osc["rho"][idx, mask],
                color=color, label=f"t = {t_arr[idx]:.1f}")

    ax.set_xlabel(r"$r \; [1/m]$")
    ax.set_ylabel(r"$\rho \; [m^2 M_{\rm Pl}^2]$")
    ax.set_title("Energy-density profiles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_paper_diagnostics(osc, use_scale_factor=True):
    """
    Produce the four main diagnostic panels from the paper:
      1) density contrast  delta_c  vs  time / ln(a)
      2) central density   rho_c    vs  time / ln(a)
      3) mass and radius   M, R     vs  time
      4) compactness       C        vs  time
    """
    t   = osc["t"]
    lna = np.log(np.maximum(osc["a"], 1e-30))
    x   = lna if use_scale_factor else t

    xlabel = r"$\ln(a)$" if use_scale_factor else r"$t \; [1/m]$"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Oscillon diagnostics  (arXiv:2304.01673 style)", fontsize=13)

    # ── Panel 1:  delta_c ─────────────────────────────────────────────────
    ax = axes[0, 0]
    positive = osc["delta_c"] > 0
    y = np.full_like(osc["delta_c"], np.nan)
    y[positive] = np.log10(osc["delta_c"][positive])
    ax.plot(x, y, "b-", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log_{10}(\delta_c)$")
    ax.set_title(r"Density contrast  $\delta_c = \rho_c / \bar\rho - 1$")
    ax.grid(True, alpha=0.3)

    # ── Panel 2:  rho_c ──────────────────────────────────────────────────
    ax = axes[0, 1]
    pos_rho = osc["rho_c"] > 0
    y2 = np.full_like(osc["rho_c"], np.nan)
    y2[pos_rho] = np.log10(osc["rho_c"][pos_rho])
    ax.plot(x, y2, "r-", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log_{10}(\rho_c \; [m^2 M_{\rm Pl}^2])$")
    ax.set_title(r"Central density  $\rho_c$")
    ax.grid(True, alpha=0.3)

    # ── Panel 3:  M and R ────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t, osc["M"], "b-", lw=1.2, label=r"$M \; [M_{\rm Pl}^2/m]$")
    ax_r = ax.twinx()
    ax_r.plot(t, osc["R"], "g--", lw=1.2, label=r"$R \; [1/m]$")
    ax.set_xlabel(r"$t \; [1/m]$")
    ax.set_ylabel(r"$M$", color="b")
    ax_r.set_ylabel(r"$R$", color="g")
    ax.set_title("Oscillon mass and radius")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4:  compactness C ──────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t, osc["C"], "k-", lw=1.2)
    ax.set_xlabel(r"$t \; [1/m]$")
    ax.set_ylabel(r"$\mathcal{C} = GM/R$")
    ax.set_title(r"Compactness  $\mathcal{C}$")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_density_contrast_comparison(osc_list, labels=None,
                                     use_scale_factor=True):
    """
    Overlay delta_c curves for several runs (e.g. different mu or lambda).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    xlabel = r"$\ln(a)$" if use_scale_factor else r"$t \; [1/m]$"

    for idx, osc in enumerate(osc_list):
        x = np.log(np.maximum(osc["a"], 1e-30)) if use_scale_factor else osc["t"]
        positive = osc["delta_c"] > 0
        y = np.full_like(osc["delta_c"], np.nan)
        y[positive] = np.log10(osc["delta_c"][positive])
        lbl = labels[idx] if labels else f"run {idx}"
        ax.plot(x, y, lw=1.2, label=lbl)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log_{10}(\delta_c)$")
    ax.set_title(r"Growth of density contrast  $\delta_c$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig, ax
