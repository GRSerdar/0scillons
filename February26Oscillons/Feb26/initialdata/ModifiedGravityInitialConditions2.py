#MGID

"""
Set the initial conditions for all the variables for an isotropic Schwarzschild BH.

See further details in https://github.com/GRChombo/engrenage/wiki/Running-the-black-hole-example.
"""

import numpy as np
from scipy.interpolate import interp1d, CubicSpline

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.tensoralgebra import *
from backgrounds.sphericalbackground import *

from matter.scalarmatter_MG import *

from initialdata.constraintsolver import *
from bssn.bssnvars import BSSNVars

def get_initial_state(grid: Grid, background, parameters, scalar_matter, bump_amplitude, R_bump, scalar_mu) :
    
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
    # Oscillon initial data Katy:

    #Importing tabulated initial data from Katy
    grr0_data   = np.loadtxt("/user/leuven/384/vsc38419/0scillons/source/initialdata/oscillaton/grr0.csv")
    lapse0_data = np.loadtxt("/user/leuven/384/vsc38419/0scillons/source/initialdata/oscillaton/lapse0.csv")
    v0_data     = np.loadtxt("/user/leuven/384/vsc38419/0scillons/source/initialdata/oscillaton/v0.csv")

    length      = np.size(grr0_data)

    # Make data symmetric in R (negative + positive)
    grr0_data   = np.concatenate((np.flip(grr0_data),   grr0_data[1:length]))
    lapse0_data = np.concatenate((np.flip(lapse0_data), lapse0_data[1:length]))
    v0_data     = np.concatenate((np.flip(v0_data),     v0_data[1:length]))

    # Areal-radius grid used by the oscillaton data
    dR = 0.01
    R  = np.linspace(-dR*(length-1), dR*(length-1), num=(length*2-1))

    # (Optional sanity check: dR < dr)
    assert dR < grid.min_dr, "dr must be >= dR of oscillaton data; use fewer grid points if needed."

    # Interpolating functions
    f_grr   = interp1d(R, grr0_data,   kind="cubic", fill_value="extrapolate")
    f_lapse = interp1d(R, lapse0_data, kind="cubic", fill_value="extrapolate")
    f_v     = interp1d(R, v0_data,     kind="cubic", fill_value="extrapolate")

    # Evaluate on the Engrenage grid radius r
    grr_profile   = f_grr(r)
    lapse_profile = f_lapse(r)
    v_profile     = f_v(r)

    #################################################################################
    # PERTURBATION ON THE OSCILLON: 

    u[:] = 0.0
    v[:] = v_profile

    # Optional: add a Gaussian bump on top of the oscillaton profile
    def bump2(r, A, Rbump):
        return A * np.exp(-(r**2) / Rbump**2)

    v[:]  +=  -bump2(r, bump_amplitude, R_bump) #Minus sign or plus sign depending on the initial profile of the bump

    #################################################################################
    # Work out metric variables (same as in oscillon id file):

    # lapse and spatial metric
    lapse[:] = f_lapse(r)
    grr = f_grr(r)
    gtt_over_r2 = 1.0
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta

    # Work out the rescaled quantities
    # Note sign error in Baumgarte eqn (2), conformal factor
    phi[:] = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
    em4phi = np.exp(-4.0*phi)
    hrr[:] = em4phi * grr - 1.0
    htt[:] = em4phi * gtt_over_r2 - 1.0
    hpp[:] = em4phi * gpp_over_r2sintheta - 1.0

    # We also calculate psi for the oscillon, so we can feed this into the
    # id solver that includes the MG corrections
    psi_osc_r = np.exp(phi)    
    
    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state)

    #################################################################################
    # Modified Gravity Changes

    GM = 0
    scalar_mass = scalar_mu
    
    # we set this to zero (we don't want an exotic spatial distribution of our initial scalar field)
    dudr = np.zeros_like(r)

    # Extra objects needed for the matter variables in modified gravity
    unflattened_state = initial_state.reshape(grid.NUM_VARS, -1)

    # Derivatives which will be needed to calculate MG terms
    d1 = grid.get_d1_metric_quantities(unflattened_state)
    d2 = grid.get_d2_metric_quantities(unflattened_state)

    # Not sure yet if I will need these
    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(unflattened_state)

    # Solve constraints
    #inflation_initial_data = CTTKBHConstraintSolver(r, GM, scalar_mass)
    inflation_initial_data = CTTKBHConstraintSolver(grid, GM, scalar_mass, parameters)
    
    ################ OSCILLON ########################
    # Interpolate psi_osc from evolution grid r to solver grid R
    cs_psi      = CubicSpline(r, psi_osc_r)
    psi_osc_R   = cs_psi(inflation_initial_data.R)
    
    # We add this to take into account the "curvature" caused by the oscillon id we import
    inflation_initial_data.set_oscillaton_background(psi_osc_R)
    ################ OSCILLON ########################

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