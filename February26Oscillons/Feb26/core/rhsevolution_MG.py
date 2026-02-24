#rhsevolution_MG.py

# python modules
import numpy as np
import time

# homemade source code
from core.grid import Grid
from bssn.tensoralgebra import *
from bssn.bssnrhs_MG import *
from bssn.bssnvars import BSSNVars
from bssn.ModifiedGravity import GBVars, get_gb_core, get_esgb_br_terms

# function that returns the rhs for each of the field vars
# see further details in https://github.com/GRChombo/engrenage/wiki/Useful-code-background
def get_rhs(t_i, current_state: np.ndarray, grid: Grid, background, matter, progress_bar, time_state, a,b, lambda_GB, coupling): 
    
    # Debugging
    #print("RHS called at t =", t_i, flush=True)

    # Set to True/False for timing and tracking progress
    # This is best tested using the BH test, where only one timestep is run
    timing_on = False
    if (timing_on) :
        start_time = time.time()
    
    ####################################################################################################
    #unpackage the state vector into the bssn vars in tensor form, and the derivatives as required

    # Just for readability and ease of indexing
    r = grid.r
    N = grid.N
    NUM_VARS = grid.NUM_VARS
    unflattened_state = current_state.reshape(NUM_VARS, -1)
    """
    state = current_state.reshape(grid.NUM_VARS, -1)
    # sanity check key fields at t=0
    if t_i == 0.0:
        # unpack in the SAME order as initial_state
        (phi, hrr, htt, hpp,
         K, arr, att, app,
         lambdar, shiftr, br, lapse,
         u, v) = state

        print("  lapse: min =", lapse.min(), "max =", lapse.max())
        print("  phi:   min =", phi.min(), "max =", phi.max())
    """
    chi0 = 0.15

    """
    if MG == True:    
        a = 0.2
        b = 0.4
        lambda_GB = 0.05
    else:
        a = 0
        b = 0
        chi0 = 0.15
        lambda_GB = 0
    """
    
    gauge_coefficients = (a, b)
    
    # First the metric vars in tensor form - see bssnvars.py
    bssn_vars = BSSNVars(N)
    bssn_vars.set_bssn_vars(unflattened_state)
    
    # get the derivatives of the bssn vars in tensor form - see bssnvars.py
    d1 = grid.get_d1_metric_quantities(unflattened_state)
    d2 = grid.get_d2_metric_quantities(unflattened_state)
    advec = grid.get_advection_d1_metric_quantities(unflattened_state, bssn_vars.shift_U)
    
    # this is where the bssn rhs will go
    bssn_rhs = BSSNVars(N)
    
    # Now set the matter
    matter.set_matter_vars(unflattened_state, bssn_vars, grid)

    if (timing_on) :    
        check_time_1 = time.time()
        print("time for set up is, ", check_time_1-start_time)
    
    #################################################################################################### 
    # impose algebraic constraints, in particular:
    # check and enforce that the determinant of \bar gamma_ij is equal to that of the hatted metric
    # (note that trace of \bar A_ij = 0 is enforced dynamically below as in Etienne
    # https://arxiv.org/abs/1712.07658v2)
    
    # work out the ratio between the bar determinant and what it should be
    determinant_bar_gamma = get_det_bar_gamma(r, bssn_vars.h_LL, background)
    determinant_hat_gamma = background.det_hat_gamma
    rescaling_factor = np.power(determinant_bar_gamma / determinant_hat_gamma, -1./3)
    
    # Check it is set correctly at first timestep
    error = np.abs(rescaling_factor - 1.0)
    if ((error > 1e-8).any() and (t_i == 0.0)) :
        print("error in rescaling factor is ", error)
        assert False, "Warning, initial det(hat gamma) != det(bar gamma), check your initial data."

    # Now enforce it
    em4phi = np.exp(-4.0*bssn_vars.phi)    

    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    gamma_UU = em4phi[:,np.newaxis,np.newaxis] *bar_gamma_UU

    new_bar_gamma_LL = rescaling_factor[:,np.newaxis,np.newaxis] * bar_gamma_LL
    bssn_vars.h_LL = (new_bar_gamma_LL - background.hat_gamma_LL) * background.inverse_scaling_matrix
        
    # Also limit the conformal factor so it doesn't blow up near BHs
    bssn_vars.phi = np.minimum(bssn_vars.phi, np.ones(N)*1.0e6)

    if (timing_on) :    
        check_time_2 = time.time()
        print("time for algebraic constraints is, ", check_time_2-check_time_1)    
    
    ####################################################################################################  
    # Structured way of calling all big objects in right order.
    ####################################################################################################

    # (1) Calculating all MG related quantities at once in an object, and then passing this object trough
    gb = GBVars(N)

    # Adding gauss bonnet terms without backreaction
    get_gb_core(gb, r, bssn_vars, d1, d2, grid, background, lambda_GB, chi0)

    # Adding the back reaction correction terms
    get_esgb_br_terms(gb, r, matter, bssn_vars, d1, d2, grid, background, lambda_GB, chi0, coupling)

    # (2) Calculating the EM tensor projections
    EMtensor = matter.get_emtensor(r, bssn_vars, background, gb)
    # Checking the runtime for matter 
    if (timing_on) :     
        check_time_3 = time.time()
        print("time for matter is, ", check_time_3-check_time_2) 

    # (3) Calculating evolution equations + returning (dudt, dvdt) to pass trough to matter_rhs
    scalar_tuple = get_bssn_rhs(bssn_rhs, r, matter, bssn_vars, d1, d2, grid, background, gb, gauge_coefficients, EMtensor)

    # (4) imports scalar tuple (dudt, dvdt) and gives them advection
    matter_rhs = matter.get_matter_rhs(r, bssn_vars, d1, background, scalar_tuple)

    ########################################################################################################
    # GAUGE EVOLUTION
    ########################################################################################################
    # Set the gauge evolution for the lapse and shift
    # eta is the 1+log slicing damping coefficient - of order 1/M_adm of spacetime

    # Modified Harmonic Gauge is implemented
    # Option 1 
    eta = 1.0
    # we are not using b_U but we keep it in to not break anything
    bssn_rhs.b_U     += 0.75 * bssn_rhs.lambda_U - eta * bssn_vars.b_U


    # ── Option 1: Modified Harmonic Gauge with Gamma driver ──
    bssn_rhs.shift_U += (0.75 * bssn_vars.lambda_U - eta * bssn_vars.shift_U
                               -((a)/(1+a)) * (0.75 * bssn_vars.lambda_U
                                               + bssn_vars.lapse[:,np.newaxis] * np.einsum("xia, xa->xi",gamma_UU, d1.lapse)))

    # We changed to K - <K> gauge for cosmology
    bssn_rhs.lapse   += - 2.0 * bssn_vars.lapse * (bssn_vars.K  - np.mean(bssn_vars.K))
    bssn_rhs.lapse   += 2*((a)/(1+a)) * bssn_vars.lapse * (bssn_vars.K  - np.mean(bssn_vars.K))


    """
    # ── Option 2: Geometric slicing (frozen lapse and shift) ──
    bssn_rhs.lapse =  np.zeros_like(r)
    bssn_rhs.shift_U = np.zeros_like(r)[:,np.newaxis]
    """
    
    ########################################################################################################
    # ADVECTION
    ########################################################################################################
    # Add advection to bssn time derivatives (this is the bit coming from the shift in the Lie derivative)
    # One sided stencils are used which helps stability
    # Note the additional advection terms from rescaling  
   
    # Scalars first
    bssn_rhs.phi   += np.einsum('xj,xj->x', background.inverse_scaling_vector * bssn_vars.shift_U, advec.phi)
    bssn_rhs.K     += np.einsum('xj,xj->x', background.inverse_scaling_vector * bssn_vars.shift_U, advec.K)
    bssn_rhs.lapse += np.einsum('xj,xj->x', background.inverse_scaling_vector * bssn_vars.shift_U, advec.lapse)

    # Vectors
    advec_lambda_U = get_vector_advection(r, bssn_vars.lambda_U, advec.lambda_U, bssn_vars.shift_U, d1.shift_U, background)
    bssn_rhs.lambda_U += advec_lambda_U
    
    # Tensors
    advec_h_LL = get_tensor_advection(r, bssn_vars.h_LL, advec.h_LL, bssn_vars.shift_U, d1.shift_U, background)
    bssn_rhs.h_LL += advec_h_LL
    
    advec_a_LL = get_tensor_advection(r, bssn_vars.a_LL, advec.a_LL, bssn_vars.shift_U, d1.shift_U, background)
    bssn_rhs.a_LL += advec_a_LL

    ########################################################################################################
    ########################################################################################################

    # Convert the tensorial forms back into the state variables, not yet flattened
    bssn_rhs_state = bssn_rhs.set_bssn_state_vars()
    
    if (matter_rhs != None) :
        rhs_state = np.concatenate([bssn_rhs_state, matter_rhs])
    else :
        rhs_state = bssn_rhs_state

    if (timing_on) :  
        check_time_4 = time.time()
        print("time for rhs is, ", check_time_4-check_time_3)    
    
    ####################################################################################################            
    # Add Kreiss Oliger dissipation which removes noise at frequency of grid resolution
    
    # kreiss-oliger damping coefficient, max_step should be limited to avoid instability
    # max sigma ~ dx / dt so dt max = dx / sigma. 
    # Since dt < 0.5 dx_min for stability anyway we can usually quite safely pick sigma = 1.0
    # but it seems to work best when weighted by the lapse and conformal factor too
    sigma = 1.0 * bssn_vars.lapse * np.exp(-2.0*bssn_vars.phi)
    
    diss = sigma * grid.get_kreiss_oliger_diss(unflattened_state)
    rhs_state += sigma * diss 
    
    if (timing_on) :    
        check_time_5 = time.time()
        print("time for matter is, ", check_time_5-check_time_4)
    
    #################################################################################################### 
    # Impose boundary conditions
    
    # overwrite outer boundaries with extrapolation (order specified in uservariables.py)
    grid.fill_outer_boundary(rhs_state)

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(rhs_state)

    if (timing_on) :    
        check_time_6 = time.time()
        print("time for boundaries is, ", check_time_6-check_time_5)    
    
    #################################################################################################### 
    # Some code for checking timing and progress output
    
    # state is a list containing last updated time t:
    # state = [last_t, dt for progress bar]
    # its values can be carried between function calls throughout the ODE integration
    last_t, deltat = time_state
    
    # call update(n) here where n = (t - last_t) / dt
    n = int((t_i - last_t)/deltat)
    progress_bar.update(n)
    # we need this to take into account that n is a rounded number:
    time_state[0] = last_t + deltat * n 

    if (timing_on) :    
        end_time = time.time()
        print("total rhs time at t= ", t_i, " is, ", end_time-start_time)
        
    #################################################################################################### 
    # Finally return the rhs, flattened into one long vector
    x=1
    """

    #### DEBUG ####
    rhs_flat = rhs_state.reshape(-1)
    rhs_norm = np.linalg.norm(rhs_flat)
    print("  ||rhs|| =", rhs_norm)
    print("  finite? ", np.all(np.isfinite(rhs_flat)))
    """
    return rhs_state.reshape(-1)
