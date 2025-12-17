"""
Set the initial conditions for all the variables for an isotropic Schwarzschild BH.

See further details in https://github.com/GRChombo/engrenage/wiki/Running-the-black-hole-example.
"""

import numpy as np

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.tensoralgebra import *
from backgrounds.sphericalbackground import *

from matter.scalarmatter_MG import *

from initialdata.constraintsolver import *
from bssn.bssnvars import BSSNVars

def get_initial_state(grid: Grid, background, parameters, scalar_matter, bump_amplitude, R, scalar_m) :
    
    assert grid.NUM_VARS == 14, "NUM_VARS not correct for bssn + scalar field"
    
    # For readability
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
    
    # we set lapse earlier, to not make MG variables blow up (initially it is zero)
    lapse.fill(1.0)

    # Extra objects needed for the matter variables in modified gravity
    unflattened_state = initial_state.reshape(grid.NUM_VARS, -1)

    # Derivatives which will be needed to calculate MG terms
    d1 = grid.get_d1_metric_quantities(unflattened_state)
    d2 = grid.get_d2_metric_quantities(unflattened_state)

    # Not sure yet if I will need these
    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(unflattened_state)
    #################################################################################


    # Set BH length scale, initial scalar data
    GM = 0.0
    scalar_mass = scalar_m
    
    # Set scalar field values
    # scalar_matter = ScalarMatter(scalar_mass)
    
    # We add a scalar bump for u 
    """
    #   Old bump function
    def bump(r, A, rl, ru):
        out = np.zeros_like(r)
        mask = (r > rl) & (r < ru)
        x = r[mask]
        out[mask] = A*(x-rl)**2*(x-ru)**2*np.exp(-1.0/(x-rl) - 1.0/(ru-x))
        return out
    """

    # This bump is centered at r=0
    def bump2(r,A,R):
        return (A * np.exp(-(r**2)/ R**2))

    # We bump the conjugate momenta of the scalar field (now at r=0)
    # The start value of the scalar field is at the end of inflation
    
    #u[:] = -0.27 + bump2(r, bump_amplitude, R)  # adding the bump makes us come closer to the minimum of the potential!

    #Trying to mimick what katy has in her paper
    u[:] = - 1.3 * 10**(-1)
    v[:] = 8 * 10**(-7)
    #u[:] += bump2(r, bump_amplitude, R)

    dudr = np.zeros_like(r)

    #################################################################################
    # Modified Gravity Changes

    # Solve constraints
    #inflation_initial_data = CTTKBHConstraintSolver(r, GM, scalar_mass)
    inflation_initial_data = CTTKBHConstraintSolver(grid, GM, scalar_mass, parameters)
    
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