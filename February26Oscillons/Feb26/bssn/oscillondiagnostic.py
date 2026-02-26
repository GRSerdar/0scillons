# oscillondiagnostic.py
#
# Oscillon diagnostics following Aurrekoetxea, Clough & Muia (arXiv:2304.01673)
#
# Computes (see paper Sec. II-C):
#   rho(r,t)   — energy-density profiles
#   rho_c(t)   — central (max) energy density
#   rho_bar(t) — proper-volume-averaged energy density
#   delta_c(t) — density contrast  rho_c / rho_bar - 1       [Eq. 17]
#   a(t)       — scale factor from volume-averaged conformal factor
#   M(t)       — oscillon mass   integral(rho sqrt(gamma) d3x) [Eq. 19]
#   V(t)       — oscillon proper volume                        [Eq. 20]
#   R(t)       — oscillon radius  (3V/4pi)^{1/3}
#   C(t)       — compactness  G M / R                          [Eq. 18]
#   K_avg(t)   — volume-averaged K  (K = -3H in FLRW limit)
#   u_c(t)     — scalar field at the center
#   u_bar(t)   — volume-averaged scalar field
#   Asq_c(t)   — central A_ij A^ij  (GW energy proxy)
#   delta_rho   — spatial profiles of  rho/rho_bar - 1
#
# IMPORTANT: pass the FULL (perturbed) evolution state, NOT a
# background-subtracted one.  The density contrast delta_c naturally
# isolates the perturbation from the background.

import numpy as np
import matplotlib.pyplot as plt

from core.grid import *
from bssn.tensoralgebra import *
from bssn.bssnvars import BSSNVars
from bssn.ModifiedGravity import GBVars, get_gb_core, get_esgb_br_terms


_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ──────────────────────────────────────────────────────────────────────────────
#  Main diagnostic function
# ──────────────────────────────────────────────────────────────────────────────

def get_oscillon_diagnostic(states_over_time, t, grid, background, matter,
                            params, surface_threshold=0.05, r_max_diag=None):
    """
    Compute oscillon diagnostics from the full evolution data.

    Parameters
    ----------
    states_over_time : ndarray, shape (num_times, NUM_VARS * N)
        Full state vector at each output time.  Must be the FULL solution
        (not background-subtracted) — physical quantities like rho, M, C
        require the actual field values.
    t : ndarray, shape (num_times,)
        Coordinate time array.
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
    r_max_diag : float or None
        Maximum radius for volume integrals (rho_bar, K_avg, u_bar,
        scale factor, oscillon mass/volume).  Set this well inside the
        outer boundary to avoid boundary-condition contamination.
        If None, the full grid is used.

    Returns
    -------
    dict with keys  (time-series have shape (num_times,),
                     profiles have shape (num_times, N)):
        t          – coordinate time array
        r          – radial grid
        rho        – energy-density profiles  rho(r, t_i)
        rho_c      – central (max) energy density
        rho_bar    – volume-averaged energy density
        delta_c    – density contrast  rho_c / rho_bar - 1       [Eq. 17]
        delta_rho  – spatial density-contrast profiles  rho/rho_bar - 1
        a          – scale factor  (from volume-averaged conformal factor)
        ln_a       – ln(a) for convenience (e-foldings since start)
        K_avg      – volume-averaged K  (= -3 H  in FLRW limit)
        M          – oscillon mass                                [Eq. 19]
        V_proper   – oscillon proper volume                       [Eq. 20]
        R          – oscillon radius  (3 V / 4 pi)^{1/3}
        C          – compactness  G M / R  with  G = 1/(8 pi)    [Eq. 18]
        u_c        – scalar field value at the center
        u_bar      – volume-averaged scalar field
        Asq_c      – central A_ij A^ij  (GW energy density proxy)
    """
    lambda_GB, a_mg, b_mg, chi0, coupling = params

    r = grid.r
    N = grid.num_points
    num_times = len(t) if np.ndim(t) > 0 else 1

    rho_all    = np.zeros((num_times, N))
    delta_rho  = np.zeros((num_times, N))
    rho_c      = np.zeros(num_times)
    rho_bar    = np.zeros(num_times)
    delta_c    = np.zeros(num_times)
    scale_fac  = np.zeros(num_times)
    K_avg_arr  = np.zeros(num_times)
    mass       = np.zeros(num_times)
    vol        = np.zeros(num_times)
    radius     = np.zeros(num_times)
    compact    = np.zeros(num_times)
    u_c_arr    = np.zeros(num_times)
    u_bar_arr  = np.zeros(num_times)
    Asq_c_arr  = np.zeros(num_times)

    r_inner = grid.min_dr * 0.5
    r_outer = r_max_diag if r_max_diag is not None else np.inf
    r_phys_mask = (r > r_inner) & (r <= r_outer)

    for i in range(num_times):
        state = states_over_time if num_times == 1 else states_over_time[i]
        state = state.reshape(grid.NUM_VARS, -1)

        # ── BSSN variables ──────────────────────────────────────────────
        bssn_N = BSSNVars(N)
        bssn_N.set_bssn_vars(state)
        matter.set_matter_vars(state, bssn_N, grid)

        d1 = grid.get_d1_metric_quantities(state)
        d2 = grid.get_d2_metric_quantities(state)

        # ── Modified-gravity objects (needed for rho_GB) ────────────────
        gb = GBVars(N)
        if lambda_GB != 0:
            get_gb_core(gb, r, bssn_N, d1, d2, grid, background,
                        lambda_GB, chi0)
            get_esgb_br_terms(gb, r, matter, bssn_N, d1, d2, grid,
                              background, lambda_GB, chi0, coupling)

        # ── Energy-momentum tensor ──────────────────────────────────────
        emtensor = matter.get_emtensor(r, bssn_N, background, gb)
        rho = emtensor.rho #+ gb.rho_GB #you can remov this if mg terms are not

        rho_all[i, :] = rho

        # ── Conformal factor / metric quantities ────────────────────────
        phi    = bssn_N.phi
        e6phi  = np.exp(6.0 * phi)
        e4phi  = np.exp(4.0 * phi)
        K_field = bssn_N.K

        # ── A_ij A^ij at center (GW energy proxy, paper p.3) ───────────
        Asquared = get_bar_A_squared(r, bssn_N, background)

        # ── Volume integrals (spherical symmetry, 4 pi r^2 factor) ─────
        # BSSN det(gamma_tilde) = det(gamma_hat) by gauge, so the
        # proper volume element is exactly  4 pi r^2 e^{6 phi} dr.
        rp    = r[r_phys_mask]
        rho_p = rho[r_phys_mask]
        e6p   = e6phi[r_phys_mask]
        e4p   = e4phi[r_phys_mask]
        K_p   = K_field[r_phys_mask]
        u_p   = matter.u[r_phys_mask]

        vol_element = 4.0 * np.pi * rp**2 * e6p
        total_vol   = _trapz(vol_element, rp)

        # Volume-averaged quantities
        rho_bar[i]    = _trapz(rho_p * vol_element, rp) / max(total_vol, 1e-30)
        K_avg_arr[i]  = _trapz(K_p   * vol_element, rp) / max(total_vol, 1e-30)
        u_bar_arr[i]  = _trapz(u_p   * vol_element, rp) / max(total_vol, 1e-30)

        # Central quantities (r = 0  or  location of max rho inside r_max_diag)
        rho_diag       = np.where(r_phys_mask, rho, -np.inf)
        rho_c[i]       = np.max(rho_diag)
        idx_center     = np.argmax(rho_diag)
        u_c_arr[i]     = matter.u[idx_center]
        Asq_c_arr[i]   = Asquared[idx_center]

        # Density contrast  [Eq. 17]
        delta_c[i] = rho_c[i] / max(rho_bar[i], 1e-30) - 1.0

        # Spatial density-contrast profiles  (for Fig. 2 style plots)
        delta_rho[i, :] = rho / max(rho_bar[i], 1e-30) - 1.0

        # Scale factor from volume-averaged conformal factor
        #   In FLRW:  e^{4 phi} = a^2,  so  a = sqrt(<e^{4 phi}>)
        # _trapz is the trapezoidal rule for numerical integration
        coord_vol = _trapz(4.0 * np.pi * rp**2, rp)
        avg_e4phi = _trapz(e4p * 4.0 * np.pi * rp**2, rp) / max(coord_vol, 1e-30)
        scale_fac[i] = np.sqrt(max(avg_e4phi, 1e-30))

        # ── Oscillon mass, volume, radius, compactness [Eqs. 18–20] ────
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
        grid.fill_inner_boundary_single_variable(delta_rho[i, :])

    # Normalize scale factor so a(t=0) = 1
    if scale_fac[0] > 0:
        scale_fac /= scale_fac[0]

    ln_a = np.log(np.maximum(scale_fac, 1e-30))

    return {
        "t"         : t,
        "r"         : r,
        "rho"       : rho_all,
        "rho_c"     : rho_c,
        "rho_bar"   : rho_bar,
        "delta_c"   : delta_c,
        "delta_rho" : delta_rho,
        "a"         : scale_fac,
        "ln_a"      : ln_a,
        "K_avg"     : K_avg_arr,
        "M"         : mass,
        "V_proper"  : vol,
        "R"         : radius,
        "C"         : compact,
        "u_c"       : u_c_arr,
        "u_bar"     : u_bar_arr,
        "Asq_c"     : Asq_c_arr,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting helpers  (mimic figures from arXiv:2304.01673)
# ──────────────────────────────────────────────────────────────────────────────

def plot_density_contrast_vs_lna(osc, ax=None, label=None, color="b"):
    """
    Paper Fig. 3 / Fig. 4 top panel:  log10(delta_c) vs ln(a).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ln_a = osc["ln_a"]
    dc   = osc["delta_c"]
    pos  = dc > 0
    y    = np.full_like(dc, np.nan)
    y[pos] = np.log10(dc[pos])

    ax.plot(ln_a, y, color=color, lw=1.2, label=label)
    ax.set_xlabel(r"$\ln(a)$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\delta_c)$", fontsize=12)
    ax.set_title(r"Density contrast  $\delta_c \equiv \rho_c / \bar\rho - 1$")
    ax.grid(True, alpha=0.3)
    return ax


def plot_central_density_vs_lna(osc, ax=None, label=None, color="r"):
    """
    Paper Fig. 4 bottom panel:  log10(rho_c / m^2 M_Pl^2) vs ln(a).
    Also shows the volume-averaged background density rho_bar for comparison.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ln_a = osc["ln_a"]
    rc   = osc["rho_c"]
    rb   = osc["rho_bar"]

    pos_c = rc > 0
    y_c   = np.full_like(rc, np.nan)
    y_c[pos_c] = np.log10(rc[pos_c])

    pos_b = rb > 0
    y_b   = np.full_like(rb, np.nan)
    y_b[pos_b] = np.log10(rb[pos_b])

    lbl_c = r"$\rho_c$" if label is None else label
    ax.plot(ln_a, y_c, color=color, lw=1.5, label=lbl_c)
    ax.plot(ln_a, y_b, color=color, lw=1.2, ls="--", alpha=0.7,label=r"$\bar{\rho}$")

    ax.set_xlabel(r"$\ln(a)$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\rho \;[m^2 M_{\rm Pl}^2])$", fontsize=12)
    ax.set_title(r"Central density $\rho_c$ vs background $\bar{\rho}$")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    return ax


def plot_density_profiles_at_times(osc, times=None, ax=None):
    """
    Radial energy-density profiles rho(r) at selected coordinate times.
    """
    if times is None:
        times = np.linspace(osc["t"][0], osc["t"][-1], 6)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    r     = osc["r"]
    t_arr = osc["t"]
    mask  = r > 0

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(times), vmax=max(times))

    for ti in times:
        idx   = np.argmin(np.abs(t_arr - ti))
        color = cmap(norm(ti))
        ax.plot(r[mask], osc["rho"][idx, mask],
                color=color, label=f"t = {t_arr[idx]:.1f}")

    ax.set_xlabel(r"$r \; [1/m]$")
    ax.set_ylabel(r"$\rho \; [m^2 M_{\rm Pl}^2]$")
    ax.set_title("Energy-density profiles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_density_contrast_profiles(osc, times=None, ax=None):
    """
    Paper Fig. 2 analogue:  radial profiles of  delta = rho/rho_bar - 1.
    """
    if times is None:
        times = np.linspace(osc["t"][0], osc["t"][-1], 6)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    r     = osc["r"]
    t_arr = osc["t"]
    mask  = r > 0

    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=min(times), vmax=max(times))

    for ti in times:
        idx   = np.argmin(np.abs(t_arr - ti))
        color = cmap(norm(ti))
        ln_a_i = osc["ln_a"][idx]
        ax.plot(r[mask], osc["delta_rho"][idx, mask],
                color=color, label=rf"$\ln(a) = {ln_a_i:.2f}$")

    ax.set_xlabel(r"$r \; [1/m]$")
    ax.set_ylabel(r"$\delta \equiv \rho / \bar\rho - 1$")
    ax.set_title(r"Density contrast profiles  (cf. paper Fig. 2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_paper_diagnostics(osc, use_scale_factor=True):
    """
    Four main diagnostic panels matching arXiv:2304.01673:
      [0,0]  log10(delta_c)  vs  ln(a)       — Fig. 3 / Fig. 4 top
      [0,1]  log10(rho_c)    vs  ln(a)       — Fig. 4 bottom
      [1,0]  M and R         vs  t (or ln a) — Fig. 5 middle/bottom
      [1,1]  C               vs  t (or ln a) — Fig. 1
    """
    t    = osc["t"]
    ln_a = osc["ln_a"]
    x    = ln_a if use_scale_factor else t

    xlabel = r"$\ln(a)$" if use_scale_factor else r"$t \; [1/m]$"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Oscillon diagnostics", fontsize=14)

    # ── Panel 1:  delta_c ─────────────────────────────────────────────────
    ax = axes[0, 0]
    pos = osc["delta_c"] > 0
    y = np.full_like(osc["delta_c"], np.nan)
    y[pos] = np.log10(osc["delta_c"][pos])
    ax.plot(x, y, "b-", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log_{10}(\delta_c)$")
    ax.set_title(r"$\delta_c = \rho_c / \bar\rho - 1$")
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
    ax.plot(x, osc["M"], "b-", lw=1.2, label=r"$M \; [M_{\rm Pl}^2/m]$")
    ax_r = ax.twinx()
    ax_r.plot(x, osc["R"], "g--", lw=1.2, label=r"$R \; [1/m]$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$M$", color="b")
    ax_r.set_ylabel(r"$R$", color="g")
    ax.set_title("Oscillon mass and radius")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 4:  compactness C ──────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(x, osc["C"], "k-", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathcal{C} = GM/R$")
    ax.set_title(r"Compactness  $\mathcal{C}$")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_density_contrast_comparison(osc_list, labels=None,
                                     use_scale_factor=True):
    """
    Overlay delta_c curves for several runs (e.g. different mu or lambda).
    Reproduces the multi-curve overlay in paper Fig. 3 / Fig. 4 top.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    xlabel = r"$\ln(a)$" if use_scale_factor else r"$t \; [1/m]$"
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(osc_list)))

    for idx, osc in enumerate(osc_list):
        x = osc["ln_a"] if use_scale_factor else osc["t"]
        positive = osc["delta_c"] > 0
        y = np.full_like(osc["delta_c"], np.nan)
        y[positive] = np.log10(osc["delta_c"][positive])
        lbl = labels[idx] if labels else f"run {idx}"
        ax.plot(x, y, lw=1.2, label=lbl, color=colors[idx])

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\delta_c)$", fontsize=12)
    ax.set_title(r"Growth of density contrast  $\delta_c$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_hubble_vs_lna(osc, ax=None, label=None, color="purple"):
    """
    Plot H(t) = -K_avg/3  vs  ln(a).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ln_a = osc["ln_a"]
    H    = -osc["K_avg"] / 3.0

    ax.plot(ln_a, H, color=color, lw=1.2, label=label)
    ax.set_xlabel(r"$\ln(a)$", fontsize=12)
    ax.set_ylabel(r"$H = -\langle K \rangle / 3 \;[m]$", fontsize=12)
    ax.set_title("Hubble parameter")
    ax.grid(True, alpha=0.3)
    return ax
