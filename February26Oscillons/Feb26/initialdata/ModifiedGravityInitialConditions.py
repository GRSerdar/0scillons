"""
Set the initial conditions for Modified Gravity + scalar field.

Perturbation spectrum follows Aurrekoetxea, Clough & Muia (arXiv:2304.01673):
  - alpha-attractor potential V(u) = m^2 mu^2 / 2 * (1 - e^{u/mu})^2
  - Vacuum fluctuations projected onto l=0 standing waves j_0(k_n r)
  - Resonance band k in [k_IR, k_UV] with k_IR = 2pi/L, k_UV = 4*k_IR, L = 64/m
  - Both field and conjugate momentum perturbations included
  - Rayleigh-distributed amplitudes matching Gaussian random field statistics
"""

import numpy as np

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.tensoralgebra import *
from backgrounds.sphericalbackground import *

from matter.scalarmatter_MG import *

from initialdata.constraintsolver import *
from bssn.bssnvars import BSSNVars


# ──────────────────────────────────────────────────────────────────────────────
#  Spherical Bessel helpers
# ──────────────────────────────────────────────────────────────────────────────

def _j0(x):
    """Spherical Bessel function j_0(x) = sin(x)/x, regular at x=0."""
    out = np.ones_like(x)
    mask = np.abs(x) > 1e-10
    out[mask] = np.sin(x[mask]) / x[mask]
    return out

def _dj0_dx(x):
    """Derivative dj_0/dx = (x cos(x) - sin(x)) / x^2, zero at x=0."""
    out = np.zeros_like(x)
    mask = np.abs(x) > 1e-10
    out[mask] = (x[mask] * np.cos(x[mask]) - np.sin(x[mask])) / x[mask]**2
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Single resonant mode perturbation (deterministic, no randomness)
# ──────────────────────────────────────────────────────────────────────────────

def single_mode_perturbation(r, A, scalar_matter, scalar_m, u_reh, v_reh,
                             k_over_m=0.2):
    """
    Excite a single l=0 standing-wave mode  j_0(k r) = sin(kr)/(kr)
    with wavenumber k inside the parametric resonance band.

    Parameters
    ----------
    r             : radial grid (code units, 1/m)
    A             : amplitude of the field perturbation max(|delta_phi|)
    scalar_matter : ScalarMatter instance (provides V, dV/du)
    scalar_m      : scalar field mass parameter m
    u_reh, v_reh  : homogeneous field and momentum at reheating
    k_over_m      : wavenumber in units of m (default 0.2, centre of
                    resonance band k/m in [0.098, 0.393])

    Returns
    -------
    delta_u, delta_v, d_delta_u_dr
    """
    k = k_over_m * scalar_m

    # V''(u_reh)
    eps = 1.0e-8
    Vpp = (scalar_matter.dVdu(u_reh + eps)
           - scalar_matter.dVdu(u_reh - eps)) / (2.0 * eps)

    omega2 = k**2 + Vpp
    omega  = np.sqrt(abs(omega2))
    omega  = max(omega, 1e-15)

    # H_reh from Friedmann
    rho_reh = 0.5 * v_reh**2 + scalar_matter.V_of_u(u_reh)
    H_reh   = np.sqrt(max(rho_reh, 0.0) / 3.0)

    # Field perturbation:  delta_phi = A * j_0(k r)
    delta_u   = A * _j0(k * r)
    d_delta_u = A * k * _dj0_dx(k * r)

    # Momentum: oscillatory part + Hubble damping
    # At t=0 the mode is a pure cosine -> dot(delta_phi) = 0 from oscillation,
    # only Hubble damping contributes
    delta_v = -H_reh * delta_u

    return delta_u, delta_v, d_delta_u

def perturbation_bump_initial_data(r, A, scalar_matter, scalar_m, u_reh, v_reh,
                                sigma_over_m=6.0):
    """
    Deterministic, localized l=0 Gaussian bump centered at r=0.

    Returns perturbations in the same format as your other initializers:
        delta_u, delta_v, d_delta_u_dr

    Conventions match your single_mode_perturbation:
    - delta_u is the field perturbation (u = u_hom + delta_u)
    - delta_v is the momentum perturbation (v = v_hom + delta_v)
    - initial oscillatory velocity is set to zero; only Hubble damping: delta_v = -H_reh * delta_u

    Parameters
    ----------
    r            : ndarray radial grid (code units, 1/m)
    A            : float   peak amplitude of the bump at r=0 (delta_u(0)=A)
    scalar_matter: ScalarMatter instance (provides V_of_u)
    scalar_m     : float   scalar mass m (sets unit conversion for sigma)
    u_reh, v_reh : floats  homogeneous field and momentum at reheating
    sigma_over_m : float   sigma in units of 1/m  (sigma = sigma_over_m / m)

    Returns
    -------
    delta_u, delta_v, d_delta_u_dr : ndarrays
    """
    # Width in code units (1/m)
    sigma = sigma_over_m / scalar_m

    # Background Hubble rate from Friedmann: H^2 = rho/3 (8πG = 1)
    rho_reh = 0.5 * v_reh**2 + scalar_matter.V_of_u(u_reh)
    H_reh   = np.sqrt(max(rho_reh, 0.0) / 3.0)

    # Field bump
    rr = r
    delta_u = A * np.exp(-(rr**2) / (2.0 * sigma**2))

    # Radial derivative (regular: derivative vanishes at r=0 automatically)
    d_delta_u = delta_u * (-(rr) / (sigma**2))

    # Momentum: "at rest" oscillatory phase + Hubble damping (same convention as your single_mode_perturbation)
    delta_v = -H_reh * delta_u

    return delta_u, delta_v, d_delta_u


# ──────────────────────────────────────────────────────────────────────────────
#  Reheating vacuum perturbation  (Aurrekoetxea, Clough & Muia 2023)
# ──────────────────────────────────────────────────────────────────────────────

def reheating_perturbation(r, A, scalar_matter, scalar_m, u_reh, v_reh,
                           seed=42):
    """
    Spherical (l=0) projection of the vacuum fluctuation spectrum at the
    start of reheating, following arXiv:2304.01673 Sec. II-A & Appendix A.

    The 3D spectrum P(k) = lambda / (2 omega_k^2) is projected onto
    standing-wave modes j_0(k_n r) in a spherical domain.  Only modes in
    the parametric-resonance band  k_IR <= k <= k_UV  are retained, with
    k_IR = 2pi/L_box,  k_UV = 4 k_IR,  L_box = 64/m  (Table I of the paper).

    Each mode has Rayleigh-distributed amplitude with sigma ~ k/omega
    (the l=0 projection of the 3D density of states times vacuum amplitude).
    Left- and right-mover decomposition gives correlated field and momentum
    perturbations.  The Hubble damping  -H_reh * delta_phi  is added to
    the momentum (Eq. A7 of the paper).

    Parameters
    ----------
    r            : radial grid (code units, 1/m)
    A            : RMS amplitude of the field perturbation  sqrt(<delta_phi^2>)
    scalar_matter: ScalarMatter instance (provides V, dV/du)
    scalar_m     : scalar field mass parameter m
    u_reh, v_reh : homogeneous field and momentum at reheating
    seed         : random seed for reproducibility

    Returns
    -------
    delta_u, delta_v, d_delta_u_dr
        Perturbations to the field, conjugate momentum, and radial field gradient.
    """
    L_box = 150.0 / scalar_m
    k_IR  = 2.0 * np.pi / L_box
    k_UV  = 4.0 * k_IR

    # V''(u_reh) from the actual potential (works for any V)
    eps  = 1.0e-8
    Vpp  = (scalar_matter.dVdu(u_reh + eps)
            - scalar_matter.dVdu(u_reh - eps)) / (2.0 * eps)

    # Hubble rate from Friedmann:  H^2 = rho / 3  (units with 8piG = 1)
    rho_reh = 0.5 * v_reh**2 + scalar_matter.V_of_u(u_reh)
    H_reh   = np.sqrt(max(rho_reh, 0.0) / 3.0)

    # Spherical box radius  =  outermost positive grid point
    r_max = np.max(np.abs(r))

    # Standing-wave wavenumbers k_n = n pi / r_max
    n_min = max(1, int(np.ceil(k_IR * r_max / np.pi)))
    n_max = max(n_min, int(np.floor(k_UV * r_max / np.pi)))

    print(n_min)
    print(n_max)

    rng = np.random.default_rng(seed)

    delta_u     = np.zeros_like(r)
    delta_v_osc = np.zeros_like(r)
    d_delta_u   = np.zeros_like(r)

    for n in range(n_min, n_max + 1):
        k_n      = n * np.pi / r_max
        omega_n2 = k_n**2 + Vpp
        omega_n  = np.sqrt(abs(omega_n2))
        omega_n  = max(omega_n, 1.0e-15)

        # Rayleigh sigma ~ k/omega  (3D density-of-states times vacuum amplitude)
        sigma = k_n / (np.sqrt(2.0) * omega_n)
        
        """
        amp_l = rng.rayleigh(sigma)
        amp_r = rng.rayleigh(sigma)
        phi_l = rng.uniform(0.0, 2.0 * np.pi)
        phi_r = rng.uniform(0.0, 2.0 * np.pi)
        """

        amp_l = amp_r = sigma*np.sqrt(np.pi/2)
        phi_l = 0.0
        phi_r = 0.0 

        # Field coefficient  Re[(a_l e^{itheta_l} + a_r e^{itheta_r}) / sqrt(2)]
        c_field = (amp_l * np.cos(phi_l)
                   + amp_r * np.cos(phi_r)) / np.sqrt(2.0)

        # Momentum coefficient  Re[i omega (...) / sqrt(2)]
        # For tachyonic modes (omega^2 < 0)  i*omega -> -|omega|, changes phase relation
        if omega_n2 >= 0:
            c_mom = -omega_n * (amp_l * np.sin(phi_l)
                                - amp_r * np.sin(phi_r)) / np.sqrt(2.0)
        else:
            c_mom = -omega_n * (amp_l * np.cos(phi_l)
                                - amp_r * np.cos(phi_r)) / np.sqrt(2.0)

        j0_kr  = _j0(k_n * r)
        dj0_kr = _dj0_dx(k_n * r)

        delta_u     += c_field * j0_kr
        delta_v_osc += c_mom   * j0_kr
        d_delta_u   += c_field * k_n * dj0_kr

    # Hubble damping of momentum (Eq. A7)
    delta_v = delta_v_osc - H_reh * delta_u

    # RMS-normalise the field perturbation to the requested amplitude A
    rms = np.sqrt(np.mean(delta_u**2))
    if rms > 0.0:
        scale        = A / rms
        delta_u     *= scale
        delta_v     *= scale
        d_delta_u   *= scale

    return delta_u, delta_v, d_delta_u


# ──────────────────────────────────────────────────────────────────────────────
#  Initial state constructor
# ──────────────────────────────────────────────────────────────────────────────

def get_initial_state(grid: Grid, background, parameters, scalar_matter,
                      bump_amplitude, R, scalar_m, u_val, v_val, seed=42,
                      k_mode=0.2):
    """
    k_mode : float or None
        If set (e.g. 0.2), use a single deterministic resonant mode
        with wavenumber k = k_mode * m.  If None, use the stochastic
        multi-mode vacuum spectrum from the paper.
    """

    assert grid.NUM_VARS == 14, "NUM_VARS not correct for bssn + scalar field"

    r = grid.r
    N = grid.num_points

    initial_state = np.zeros((grid.NUM_VARS, N))
    (
        phi,
        hrr,
        htt,
        hpp,
        K,
        arr,
        att,
        app,
        lambdar,
        shiftr,
        br,
        lapse,
        u,
        v
    ) = initial_state

    #################################################################################
    # Modified Gravity Changes

    lapse.fill(1.0)
    shiftr.fill(0.0)

    unflattened_state = initial_state.reshape(grid.NUM_VARS, -1)

    d1 = grid.get_d1_metric_quantities(unflattened_state)
    d2 = grid.get_d2_metric_quantities(unflattened_state)

    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(unflattened_state)
    #################################################################################

    GM = 0.0

    # Homogeneous field values at the start of reheating
    u[:] = u_val
    v[:] = v_val


    # This bump is centered at r=0
    def bump2(r,A,R):
        return (A * np.exp(-(r**2)/ (2*R**2)))
    
    # Analytic derivative of the bump (old version had two bugs: operator precedence and wrong coefficient)
    # def dbump2_dr(r, A, R):
    #     return A * np.exp(-(r**2)/2*R**2) * (-2*r/R**2)
    def dbump2_dr(r, A, R):
        return A * np.exp(-(r**2) / (2*R**2)) * (-r / R**2)
    
    """
    if bump_amplitude > 0.0:
        if k_mode is not None:
            delta_u, delta_v, dudr = perturbation_bump_initial_data(r, A=1e-4, scalar_matter=scalar_matter, scalar_m=scalar_m, u_reh=u_val, v_reh=v_val, sigma_over_m=6.0)
        else:
            delta_u, delta_v, dudr = reheating_perturbation(
                r, bump_amplitude, scalar_matter, scalar_m, u_val, v_val,
                seed=seed,
            )
        u[:] += delta_u
        v[:] += delta_v
    """
    # if bump_amplitude > 0.0:
    if bump_amplitude != 0.0:
        u[:] += bump2(r, bump_amplitude, R)
        #v[:] += -bump2(r, bump_amplitude, R) * 0.1
        dudr = dbump2_dr(r, bump_amplitude, R)
    else:
        dudr = np.zeros_like(r)

    #################################################################################
    # Modified Gravity Changes

    # Solve constraints
    #inflation_initial_data = CTTKBHConstraintSolver(r, GM, scalar_mass)
    inflation_initial_data = CTTKBHConstraintSolver(grid, GM, scalar_m, parameters)
    
    # setting the matter variables 
    scalar_matter.set_matter_vars(unflattened_state, bssn_vars, grid)

    # setting the matter source 
    inflation_initial_data.set_matter_source(u, v, dudr, d1,d2 ,scalar_matter, bssn_vars, background, grid)
    
    psi4, K[:], arr[:], att[:], app[:] = inflation_initial_data.get_evolution_vars() 

    #################################################################################
    # set non zero metric values
    grr = psi4
    gtt_over_r2 = grr
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta
    
    # Note sign error in Baumgarte eqn (2), conformal factor
    phi[:] = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
    # Cap the phi value in the centre to stop unphysically large numbers at singularity
    phi[:] = np.clip(phi, None, 10.0)
    em4phi = np.exp(-4.0*phi)
    hrr[:] = em4phi * grr - 1.0
    htt[:] = em4phi * gtt_over_r2 - 1.0
    hpp[:] = em4phi * gpp_over_r2sintheta - 1.0 

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state)
    
    # Set up matrices
    zeros = np.zeros_like(hrr)
    h_LL = np.array([[hrr, zeros, zeros],[zeros, htt, zeros],[zeros, zeros, hpp]])
    h_LL = np.moveaxis(h_LL, -1, 0) 
    first_derivative_indices = [idx_hrr, idx_htt, idx_hpp]
    dstate_dr = grid.get_first_derivative(initial_state, first_derivative_indices)
    (dhrr_dr, dhtt_dr, dhpp_dr) = dstate_dr[first_derivative_indices]
        
    # This is d h_ij / dx^k = dh_dx[x,i,j,k]
    d1_h_dx = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
    d1_h_dx[:,i_r,i_r, i_r]  = dhrr_dr
    d1_h_dx[:,i_t,i_t, i_r]  = dhtt_dr
    d1_h_dx[:,i_p,i_p, i_r]  = dhpp_dr
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, h_LL, d1_h_dx, background)
    lambdar[:]   = Delta_U[:,i_r]

    # Fill boundary cells for lambdar
    grid.fill_outer_boundary(initial_state, [idx_lambdar])

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state, [idx_lambdar])
            
    return initial_state.reshape(-1)