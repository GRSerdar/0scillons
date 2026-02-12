# oscillondiagnostic.py
#
# Oscillon diagnostic for 1D spherical symmetry (Engrenage-style NR code),
# following the paper's definitions:
#
#  1) Central density contrast (as in the paper):
#       delta_c = rho_c / rho_bar - 1
#     where rho_bar is the proper-volume weighted mean density on the slice.
#
#  2) Object (oscillon) region Omega (THIS is the key difference vs my earlier version):
#       Omega = { x^i : rho(x^i) / rho_osc^c > f_cut }
#     i.e. rho(r) > f_cut * rho_c, where rho_c is the central (peak) density of the oscillon.
#     The paper uses f_cut = 0.05 (5%).
#
#     In 1D spherical symmetry, "the region connected to the center" is simply
#     the contiguous interval from the first physical point outward until the first
#     time rho drops below f_cut*rho_c.
#
#  3) Mass and compactness (as in the paper):
#       M = ∫_Omega rho dV
#       R = areal radius at the boundary of Omega: R = sqrt(gamma_{θθ}(r_s))
#       C = G M / R
#
# NOTES:
# - rho is taken directly from matter.get_emtensor(...).rho (Eulerian energy density).
# - All integrals use the proper 3-volume element on the slice:
#       dV = 4π * sqrt(gamma_rr) * gamma_{θθ} dr
# - Ghost zones are excluded by default.
#
# Copy/paste as a new file alongside constraintsdiagnostic.py

import numpy as np
import matplotlib.pyplot as plt

from core.grid import Grid
from core.spacing import NUM_GHOSTS

from bssn.bssnvars import BSSNVars
from bssn.tensoralgebra import get_bar_gamma_LL
from bssn.ModifiedGravity import GBVars, get_gb_core, get_esgb_br_terms


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _iter_states_over_time(states_over_time, t, grid: Grid):
    """
    Yield (state_2d, t_i) pairs.

    states_over_time can be:
      - shape (NUM_VARS*N,)           for a single time
      - shape (num_times, NUM_VARS*N) for many times

    t can be:
      - scalar
      - array-like length num_times
    """
    states = np.asarray(states_over_time)

    # Determine number of times
    if states.ndim == 1:
        num_times = 1
    else:
        num_times = states.shape[0]

    # Normalize t to array of length num_times
    t_arr = np.asarray(t)
    if num_times == 1:
        if t_arr.size == 1:
            t_arr = np.array([float(t_arr)], dtype=float)
        else:
            t_arr = np.array([float(t_arr[0])], dtype=float)
    else:
        t_arr = t_arr.astype(float)

    for i in range(num_times):
        if num_times == 1:
            state_i = states
        else:
            state_i = states[i]

        yield state_i.reshape(grid.NUM_VARS, -1), float(t_arr[i])


def _linear_interp(x, xgrid, ygrid):
    """
    Linear interpolation y(x) on a monotonic grid xgrid.
    """
    if x <= xgrid[0]:
        return float(ygrid[0])
    if x >= xgrid[-1]:
        return float(ygrid[-1])

    j = int(np.searchsorted(xgrid, x))  # first index with xgrid[j] >= x
    if j <= 0:
        return float(ygrid[0])

    x0 = float(xgrid[j - 1])
    x1 = float(xgrid[j])
    y0 = float(ygrid[j - 1])
    y1 = float(ygrid[j])

    w = (x - x0) / (x1 - x0)
    return (1.0 - w) * y0 + w * y1


def _find_surface_by_fraction_of_central_density(r, rho, rho_c, f_cut, sl):
    """
    Paper definition:
        Omega = { r : rho(r) / rho_c > f_cut }  connected to center.

    In 1D spherical symmetry:
      - start at first physical point (sl.start) and march outward
      - stop at the first downward crossing of rho = f_cut * rho_c

    Returns
    -------
    r_s : float
        Coordinate radius of the surface
    i_last : int
        Last grid index (global) still inside Omega
    """
    if rho_c <= 0.0 or (not np.isfinite(rho_c)):
        # Degenerate / unphysical; no meaningful surface
        i0 = sl.start or 0
        return float(r[i0]), int(i0)

    thr = float(f_cut) * float(rho_c)

    r_phys = r[sl]
    rho_phys = rho[sl]

    if r_phys.size == 0:
        return float(r[0]), 0

    inside = rho_phys > thr

    # Since thr is a fraction of rho_c, inside[0] should normally be True.
    if not inside[0]:
        i0 = sl.start or 0
        return float(r[i0]), int(i0)

    # March outward while inside
    k = 0
    while (k + 1) < inside.size and inside[k + 1]:
        k += 1

    i_last = (sl.start or 0) + k

    # If never exits, surface is outer edge of physical domain
    if k == inside.size - 1:
        return float(r_phys[-1]), int(i_last)

    # Interpolate crossing between k (inside) and k+1 (outside)
    r0 = float(r_phys[k])
    r1 = float(r_phys[k + 1])
    y0 = float(rho_phys[k] - thr)       # > 0
    y1 = float(rho_phys[k + 1] - thr)   # <= 0

    if y0 == y1:
        r_s = r0
    else:
        r_s = r0 + (r1 - r0) * (y0 / (y0 - y1))

    return float(r_s), int(i_last)


# --------------------------------------------------------------------------------------
# Main diagnostic
# --------------------------------------------------------------------------------------

def get_oscillon_diagnostic(
    states_over_time,
    t,
    grid: Grid,
    background,
    matter,
    params,
    f_cut=0.05,
    G=1.0,
    r_mean_max=None,
    exclude_ghosts=True,
    make_plots=False,
    debug=False,
):
    """
    Compute oscillon diagnostics over time, matching the paper.

    Parameters
    ----------
    params : (lambda_GB, a, b, chi0, coupling)
        Same tuple as constraintsdiagnostic.py.
    f_cut : float
        Paper cut fraction, default 0.05 (5%): rho / rho_c > f_cut defines Omega.
    G : float
        Gravitational constant in code units (often 1).
    r_mean_max : float or None
        If not None, compute rho_bar only over r <= r_mean_max (still proper-volume weighted).
        This only affects delta_c, not Omega (Omega is based on rho_c).
    exclude_ghosts : bool
        If True, exclude ghost zones for integrals and for finding the surface.
    debug : bool
        If True, prints quick info each time slice.

    Returns
    -------
    out : dict
        Arrays of length num_times:
          t, rho_c, rho_bar, delta_c, r_s, M, R, C
        Plus rho_profiles, r, and optional matplotlib figs.
    """
    lambda_GB, a, b, chi0, coupling = params

    r = grid.r
    N = grid.N

    # Physical slice (exclude ghost zones)
    if exclude_ghosts:
        sl = slice(NUM_GHOSTS, -NUM_GHOSTS)
        if (N <= 2 * NUM_GHOSTS) or (r[sl].size == 0):
            sl = slice(0, N)
    else:
        sl = slice(0, N)

    i0 = sl.start or 0  # first physical point index

    # Mean-density window mask (only used for rho_bar)
    if r_mean_max is None:
        use_window = None
    else:
        use_window = (r <= float(r_mean_max))

    # Determine num_times for allocation
    states = np.asarray(states_over_time)
    if states.ndim == 1:
        num_times = 1
    else:
        num_times = states.shape[0]

    # Allocate outputs
    tt = np.zeros(num_times)

    rho_c_arr = np.zeros(num_times)
    rho_bar_arr = np.zeros(num_times)
    delta_c_arr = np.zeros(num_times)

    r_s_arr = np.zeros(num_times)
    M_arr = np.zeros(num_times)
    R_arr = np.zeros(num_times)
    C_arr = np.zeros(num_times)

    rho_profiles = np.zeros((num_times, N))

    # Loop over time
    for itime, (state, t_i) in enumerate(_iter_states_over_time(states_over_time, t, grid)):
        tt[itime] = t_i

        # Unpack fields
        bssn_vars = BSSNVars(N)
        bssn_vars.set_bssn_vars(state)

        matter.set_matter_vars(state, bssn_vars, grid)

        # Derivatives required for MG objects
        d1 = grid.get_d1_metric_quantities(state)
        d2 = grid.get_d2_metric_quantities(state)

        # MG variables
        gb = GBVars(N)
        get_gb_core(gb, r, bssn_vars, d1, d2, grid, background, lambda_GB, chi0)
        get_esgb_br_terms(gb, r, matter, bssn_vars, d1, d2, grid, background, lambda_GB, chi0, coupling)

        # Matter energy density (Eulerian)
        em = matter.get_emtensor(r, bssn_vars, background, gb)
        rho = np.asarray(em.rho)
        rho_profiles[itime, :] = rho

        # --- Build physical spatial metric gamma_LL = e^{4phi} * bar_gamma_LL ---
        em4phi = np.exp(-4.0 * bssn_vars.phi)
        e4phi = 1.0 / em4phi

        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL

        # Indices are (r,theta,phi)=(0,1,2) in your background
        gamma_rr = gamma_LL[:, 0, 0]
        gamma_thth = gamma_LL[:, 1, 1]

        # Areal radius: R(r) = sqrt(gamma_{θθ})
        R_areal = np.sqrt(np.maximum(gamma_thth, 0.0))

        # Proper volume weight after integrating angles:
        #   dV = 4π * sqrt(gamma_rr) * gamma_{θθ} dr
        w = 4.0 * np.pi * np.sqrt(np.maximum(gamma_rr, 0.0)) * np.maximum(gamma_thth, 0.0)

        # --- Central density rho_c ---
        rho_c = float(rho[i0])
        rho_c_arr[itime] = rho_c

        # --- Mean density rho_bar (for delta_c only) ---
        if use_window is None:
            r_mean = r[sl]
            w_mean = w[sl]
            rho_mean = rho[sl]
        else:
            idx = np.arange(N)
            inside_phys = (idx >= (sl.start or 0)) & (idx < (sl.stop if sl.stop is not None else N))
            use = inside_phys & use_window
            if not np.any(use):
                use = inside_phys
            r_mean = r[use]
            w_mean = w[use]
            rho_mean = rho[use]

        V_box = np.trapz(w_mean, r_mean)
        if (not np.isfinite(V_box)) or (V_box <= 0.0):
            V_box = 1.0

        rho_bar = np.trapz(rho_mean * w_mean, r_mean) / V_box
        rho_bar_arr[itime] = rho_bar

        if rho_bar != 0.0:
            delta_c_arr[itime] = rho_c / rho_bar - 1.0
        else:
            delta_c_arr[itime] = np.nan

        # --- Paper surface definition: rho(r) / rho_c > f_cut ---
        r_s, i_last = _find_surface_by_fraction_of_central_density(r, rho, rho_c, f_cut, sl)
        r_s_arr[itime] = r_s

        # --- Mass inside Omega: M = ∫_0^{r_s} rho dV ---
        r0_val = float(r[i0])

        if r_s <= r0_val:
            M = 0.0
        else:
            # integrate on points i0..i_last
            r_seg = r[i0 : i_last + 1]
            w_seg = w[i0 : i_last + 1]
            rho_seg = rho[i0 : i_last + 1]

            M = np.trapz(rho_seg * w_seg, r_seg)

            # partial cell correction from r[i_last] to r_s
            phys_stop = sl.stop if sl.stop is not None else N
            if (i_last + 1) < phys_stop and r_s > float(r[i_last]):
                rho_rs = _linear_interp(r_s, r, rho)
                w_rs = _linear_interp(r_s, r, w)
                M += 0.5 * ((rho[i_last] * w[i_last]) + (rho_rs * w_rs)) * (r_s - float(r[i_last]))

        M_arr[itime] = float(M)

        # --- Areal radius at the surface ---
        R_s = _linear_interp(r_s, r, R_areal)
        R_arr[itime] = float(R_s)

        # --- Compactness ---
        if R_s > 0.0:
            C_arr[itime] = float(G * M / R_s)
        else:
            C_arr[itime] = np.nan

        if debug:
            thr = f_cut * rho_c
            print(
                f"[t={t_i:.6g}] rho_c={rho_c:.6e}  rho_bar={rho_bar:.6e}  "
                f"delta_c={delta_c_arr[itime]:+.3e}  thr={thr:.6e}  r_s={r_s:.6g}  "
                f"M={M:.6e}  R={R_s:.6e}  C={C_arr[itime]:.6e}"
            )

    out = {
        "t": tt,
        "rho_c": rho_c_arr,
        "rho_bar": rho_bar_arr,
        "delta_c": delta_c_arr,
        "r_s": r_s_arr,
        "M": M_arr,
        "R": R_arr,
        "C": C_arr,
        "rho_profiles": rho_profiles,
        "r": r,
        "f_cut": float(f_cut),
        "r_mean_max": r_mean_max,
        "exclude_ghosts": bool(exclude_ghosts),
    }

    # ----------------------------------------------------------------------------------
    # Optional plots
    # ----------------------------------------------------------------------------------
    if make_plots:
        fig1, ax1 = plt.subplots()
        ax1.plot(tt, delta_c_arr)
        ax1.set_xlabel("t")
        ax1.set_ylabel(r"$\delta_c=\rho_c/\bar{\rho}-1$")
        ax1.set_title("Central density contrast")
        ax1.grid(True)

        fig2, ax2 = plt.subplots()
        ax2.plot(tt, C_arr)
        ax2.set_xlabel("t")
        ax2.set_ylabel(r"$C = GM/R$")
        ax2.set_title("Oscillon compactness")
        ax2.grid(True)

        fig3, ax3 = plt.subplots()
        ax3.plot(tt, M_arr, label="M")
        ax3.plot(tt, R_arr, label="R_areal(surface)")
        ax3.set_xlabel("t")
        ax3.set_title("Mass and areal radius (paper-defined Omega)")
        ax3.legend()
        ax3.grid(True)

        out["fig_delta_c"] = fig1
        out["fig_compactness"] = fig2
        out["fig_M_R"] = fig3

    return out


# --------------------------------------------------------------------------------------
# Convenience plot: density profiles at chosen times
# --------------------------------------------------------------------------------------

def plot_density_profiles_at_times(out, times, max_profiles=6):
    """
    Plot rho(r) at selected times with the paper threshold (dashed):
        rho = f_cut * rho_c(t)

    out : dict returned by get_oscillon_diagnostic
    times : list of times to plot (nearest slice is used)
    """
    r = out["r"]
    t = out["t"]
    rho_profiles = out["rho_profiles"]
    rho_c = out["rho_c"]
    f_cut = out["f_cut"]

    times = list(times)[:max_profiles]

    fig, ax = plt.subplots()

    for tv in times:
        k = int(np.argmin(np.abs(t - tv)))
        rho = rho_profiles[k]
        thr = f_cut * rho_c[k]

        ax.plot(r, rho, label=f"t={t[k]:.6g}")
        ax.axhline(thr, linestyle="--", linewidth=1)

    ax.set_xlabel("r (coordinate)")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("Density profiles (dashed = $f_{cut}\\,\\rho_c$)")
    ax.legend()
    ax.grid(True)

    return fig
