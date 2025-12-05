# ModifiedGravity.py 

import numpy as np

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.bssnvars import *
from bssn.tensoralgebra import *

# -----------------------------
# Small container, like bssn_rhs
# -----------------------------
class GBVars:
    """
    Holds all Gauss-Bonnet related objects
    """
    def __init__(self, N):
        # Core geometry
        self.bar_L_GB   = np.zeros(N)           # scalar
        self.M_LL       = np.zeros((N,3,3))     # tensor (lower-lower)
        self.Trace_M    = np.zeros(N)           # scalar
        self.TraceFree_M_LL = np.zeros((N,3,3)) # tensor
        self.TraceFree_M_UU = np.zeros((N,3,3)) # tensor
        self.N_L        = np.zeros((N,3))       # vector (lower)
        self.gamma_LL   = np.zeros((N,3,3))
        self.gamma_UU   = np.zeros((N,3,3))
        self.bar_A_LL   = np.zeros((N,3,3))

        # Matter-coupled “barred” ESGB BR terms
        self.rho_GB                 = np.zeros(N)
        self.S_GB_L                 = np.zeros((N,3))
        self.bar_TraceFree_S_GB_LL  = np.zeros((N,3,3))
        self.bar_S_GB               = np.zeros(N)
        self.Omega_LL               = np.zeros((N,3,3))
        self.d1Lambdadu             = np.zeros(N)  # scalar “coupling-derivative” factor
        self.d2Lambdadduu           = np.zeros(N)  # scalar “coupling-derivative” factor

# -------------------------------------------------
# Core GB geometry 
# -------------------------------------------------

def get_gb_core(gb_vars: GBVars, r, bssn_vars, d1, d2, grid, background,lambda_GB, chi0):
    """
    Calculates all GB objects that will also be used in other objects in other files 
    - bssn_rhs_MG.py (evolution equation dKdt and dLambdadt will be modified due to Llibert gauge)
    - rhsevolution_MG.py (for the gauge)
    - scalarmatter_MG.py (calculating corrections to rho, Si, ...)
    """
    one_two = 1.0/2.0
    one_sixth = 1.0/6.0
    one_third = 1.0/3.0
    two_thirds = 2.0/3.0
    four_thirds = 4.0/3.0
    two_nine = 2.0/9.0
    third_two = 3.0/2.0

    ################### BSSN variables ######################################

    em4phi = np.exp(-4.0 * bssn_vars.phi)
    e4phi  = 1.0/em4phi
    ilapse = 1.0/(bssn_vars.lapse)
    K = bssn_vars.K

    # Conformal metrics
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    gamma_UU = em4phi[:,np.newaxis,np.newaxis] *bar_gamma_UU
    gamma_LL = e4phi[:,np.newaxis,np.newaxis] * bar_gamma_LL

    # \bar A_ij and useful contractions
    bar_A_LL = get_bar_A_LL(r, bssn_vars, background)

    # Connections & Christoffels
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    # Shift related objects
    shift_U = bssn_vars.shift_U #scaled shift (lower case)
    Shift_U = background.inverse_scaling_vector * shift_U #Captial Shift

    d1_Shift_U = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,np.newaxis]  
                     + d1.shift_U * background.inverse_scaling_vector[:,:,np.newaxis]) 
    
    div_shift = np.einsum('xii->x', d1_Shift_U) + np.einsum('xiij,xj->x', bar_chris, Shift_U)

    # Derivatives of A_LL
    s_times_d1_a = background.scaling_matrix[:,:,:,np.newaxis] * d1.a_LL
    a_times_d1_s = bssn_vars.a_LL[:,:,:,np.newaxis] * background.d1_scaling_matrix
    # To deal with the scaling matrix indices being jki in stead of ijk 
    a_times_d1_s = np.moveaxis(a_times_d1_s, 3, 1)   # xbca -> xabc
    s_times_d1_a = np.moveaxis(s_times_d1_a, 3, 1)   # xbca -> xabc
    
    ################### BSSN variables ######################################

    # Conformal Ricci
    bar_Rij = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL,bssn_vars.lambda_U, d1.lambda_U,
                                   Delta_U, Delta_ULL, Delta_LLL,bar_gamma_UU, bar_gamma_LL, background)
    
    # Physical Ricci 
    Rij = (bar_Rij 
           - 2*d2.phi
           + 2*np.einsum('xlij,xl->xij', bar_chris, d1.phi)
           - 2*np.einsum('xij,xlm,xlm->xij',bar_gamma_LL, bar_gamma_UU, d2.phi)
           + 2*np.einsum('xij, xlm, xklm, xk->xij', bar_gamma_LL, bar_gamma_UU, bar_chris, d1.phi)
           + 4*np.einsum('xi, xj->xij', d1.phi, d1.phi)
           - 4*np.einsum('xij, xlm, xl, xm->xij', bar_gamma_LL, bar_gamma_UU, d1.phi, d1.phi))

    # Aik Ajk
    AikAjk = np.einsum('xik, xkb, xjb->xij', bar_A_LL, bar_gamma_UU, bar_A_LL)

    # M_ij
    M_LL = (Rij
           + e4phi[:, np.newaxis, np.newaxis] *(two_nine * bar_gamma_LL* bssn_vars.K[:, np.newaxis, np.newaxis] * bssn_vars.K[:, np.newaxis, np.newaxis]
                    + one_third * bssn_vars.K[:, np.newaxis, np.newaxis] * bar_A_LL - AikAjk)) 

    Trace_M = get_trace(M_LL, gamma_UU)

    TraceFree_M_LL = M_LL - one_third * gamma_LL * Trace_M[:, np.newaxis, np.newaxis]
    TraceFree_M_UU = np.einsum('xia,xjb,xab->xij', gamma_UU, gamma_UU, TraceFree_M_LL)

    N_L = (np.einsum("xjm, xjim->xi",bar_gamma_UU, a_times_d1_s)
           + np.einsum("xjm, xjim->xi", bar_gamma_UU, s_times_d1_a)
           - np.einsum("xjm, xkji, xkm->xi", bar_gamma_UU, bar_chris, bar_A_LL)
           - np.einsum("xjm, xkjm, xik->xi", bar_gamma_UU, bar_chris, bar_A_LL) 
           + 6 * np.einsum("xj, xjb, xib->xi",d1.phi, bar_gamma_UU, bar_A_LL)
           - two_thirds * d1.K)
    
    N_squared = np.einsum('xi, xai, xa->x', N_L,gamma_UU ,N_L)

    # Pieces for barred L_GB (no time derivs of K, A_ij)
    D2_lapse = em4phi*(np.einsum('xij,xij->x', bar_gamma_UU, d2.lapse)
                  - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.lapse)
                  + 2.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.lapse, d1.phi))
    
    Asquared = get_bar_A_squared(r, bssn_vars, background)
    
    # Line1
    # We put the \partial_\perp terms to zero
    Line1 = (-four_thirds*Trace_M* (ilapse * 0 # The zero represents the dkdt term
                                    + ilapse * D2_lapse 
                                    + Asquared 
                                    - bssn_vars.K * bssn_vars.K))

    # bar A_kj bar A^j_l
    bar_A_LL_A_UL = np.einsum("xaj,xkj,xal->xkl",bar_gamma_UU,bar_A_LL,bar_A_LL)

    # D_k D_l lapse 
    DkDl_lapse = (d2.lapse
                  - np.einsum('xkij,xk->xij', bar_chris, d1.lapse)
                  - 2.0 * np.einsum('xi,xj->xij', d1.phi, d1.lapse)
                  - 2.0 * np.einsum('xj,xi->xij', d1.phi, d1.lapse))

    # Line2 
    # We put the \partial_\perp terms to zero
    """
    Line2 = (8*( ilapse * np.einsum("xkl, xkl->x", TraceFree_M_UU, DkDl_lapse)
                +  e4phi*( np.einsum("xkl,xkl->x",TraceFree_M_UU,bar_A_LL_A_UL)
                         - two_thirds*(bssn_vars.K - ilapse * div_shift)*np.einsum("xkl,xkl->x",TraceFree_M_UU,bar_A_LL))))
    """
    # Copy of line 2 where divshift is not a full covariant derivative 
    Line2 = (8*( ilapse * np.einsum("xkl, xkl->x", TraceFree_M_UU, DkDl_lapse)
                +  e4phi*( np.einsum("xkl,xkl->x",TraceFree_M_UU,bar_A_LL_A_UL)
                         - two_thirds*(bssn_vars.K - ilapse * div_shift )*np.einsum("xkl,xkl->x",TraceFree_M_UU,bar_A_LL))))

    # D_L A_LL and raised version
    D_L_A_LL = e4phi[:, np.newaxis, np.newaxis, np.newaxis]*(  a_times_d1_s
                                                             + s_times_d1_a
                                                             - np.einsum("xmij, xmk->xijk",bar_chris, bar_A_LL)
                                                             - np.einsum("xmik, xjm->xijk",bar_chris, bar_A_LL)
                                                             - 2*np.einsum("xj, xik->xijk",d1.phi, bar_A_LL)
                                                             - 2*np.einsum("xk, xji->xijk",d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xij, xml, xl, xmk->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xik, xml, xl, xjm->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL))
    
    # D^i A^jk
    D_U_A_UU = np.einsum('xai,xbj,xkc,xabc->xijk', gamma_UU, gamma_UU, gamma_UU, D_L_A_LL)
    # D_i A_jk * D^[j A^i]k
    Product = ( np.einsum('xijk,xijk->x', D_L_A_LL, D_U_A_UU) - np.einsum('xijk,xjik->x', D_L_A_LL, D_U_A_UU))
    # D_i K N^i
    D_L_K_N_U   = np.einsum('xai,xi,xa->x', gamma_UU, d1.K, N_L)
    # D_i K D_i K
    D_L_K_D_U_K = np.einsum('xai,xi,xa->x', gamma_UU, d1.K, d1.K)

    Line3 = -4*(2*(Product)-four_thirds*D_L_K_N_U - four_thirds*one_third*D_L_K_D_U_K-2*N_squared)

    bar_L_GB =  Line1 + Line2 + Line3

    # Coupling functions
    # (S function)
    
    
    """ 
    # Linear coupling (old)
    d1Lambdadu = lambda_GB * S
    d2Lambdadduu = 0.0 * S 
    """

    # Store everything in gb_vars
    gb_vars.gamma_LL[:]       = gamma_LL
    gb_vars.gamma_UU[:]       = gamma_UU
    gb_vars.bar_A_LL[:]       = bar_A_LL
    gb_vars.bar_L_GB[:]       = bar_L_GB
    gb_vars.M_LL[:]           = M_LL
    gb_vars.Trace_M[:]        = Trace_M
    gb_vars.TraceFree_M_LL[:] = TraceFree_M_LL
    gb_vars.TraceFree_M_UU[:] = TraceFree_M_UU
    gb_vars.N_L[:]            = N_L
    #gb_vars.d1Lambdadu[:]     = d1Lambdadu
    #gb_vars.d2Lambdadduu[:]   = d2Lambdadduu

# ------------------------------------------------------------
# ESGB backreaction terms
# ------------------------------------------------------------
def get_esgb_br_terms(gb_vars: GBVars, r, matter, bssn_vars, d1, d2, grid, background,
                      lambda_GB, chi0, coupling):
    """
    Calculates all backreacdtion contributions 
    The barred variables in this function like bar_F, bar_F_LL, bar_S_GB and bar_S_GB_LL do not mean conformal contractions
    But rather not involving time derivatives of pape A or pape K
    I agree the "barred" notation can cause confusion 
    """
    
    assert getattr(matter, "matter_vars_set", False), "Matter vars not set (call matter.set_matter_vars(...) first)."
    
    one_sixth   = 1.0/6.0
    one_third   = 1.0/3.0
    two_thirds  = 2.0/3.0
    four_thirds = 4.0/3.0
    two_ninths  = 2.0/9.0
    three_halfs = 3.0/2.0

    # Shorthands from gb_vars (already filled by get_gb_core)
    gamma_UU = gb_vars.gamma_UU
    gamma_LL = gb_vars.gamma_LL
    bar_A_LL = gb_vars.bar_A_LL
    M_LL     = gb_vars.M_LL
    Trace_M  = gb_vars.Trace_M
    TF_M_LL  = gb_vars.TraceFree_M_LL
    TF_M_UU  = gb_vars.TraceFree_M_UU
    N_L      = gb_vars.N_L
    bar_L_GB = gb_vars.bar_L_GB
    #d1Lambdadu = gb_vars.d1Lambdadu
    #d2Lambdadduu = gb_vars.d2Lambdadduu

    # Basic factors
    em4phi = np.exp(-4.0*bssn_vars.phi)
    e4phi  = 1.0/em4phi
    ilapse = 1.0/bssn_vars.lapse
    Asquared = get_bar_A_squared(r, bssn_vars, background)

    ####### Coupling functions ########
    chi = em4phi
    S   = 1.0/(1.0 + np.exp(-100.0*(chi - chi0)))


    if coupling == "linear":
        #Linear coupling
        d1Lambdadu = lambda_GB * S
        d2Lambdadduu = 0 * S

        gb_vars.d1Lambdadu[:]     = d1Lambdadu
        gb_vars.d2Lambdadduu[:]   = d2Lambdadduu
    
    elif coupling == "quadratic": # Gaussian
        beta = 250

        # Quadratic coupling
        d1Lambdadu = S*lambda_GB* (2.0 *matter.u * np.exp(-beta * matter.u * matter.u))
        d2Lambdadduu = S*lambda_GB* ( 2* np.exp(- beta * matter.u * matter.u )* (1-2*beta*(matter.u * matter.u))) 

        gb_vars.d1Lambdadu[:]     = d1Lambdadu
        gb_vars.d2Lambdadduu[:]   = d2Lambdadduu
    
    ###################################

    # Conformal metrics (needed for a few contractions)
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    # Christoffels + helpers 
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    # Shift related objects
    shift_U = bssn_vars.shift_U #scaled shift (lower case)
    Shift_U = background.inverse_scaling_vector * shift_U #Captial Shift
    d1_Shift_U = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,np.newaxis]  
                     + d1.shift_U * background.inverse_scaling_vector[:,:,np.newaxis]) 
    div_shift = np.einsum('xii->x', d1_Shift_U) + np.einsum('xiij,xj->x', bar_chris, Shift_U)

    s_times_d1_a = background.scaling_matrix[:,:,:,np.newaxis] * d1.a_LL
    a_times_d1_s = bssn_vars.a_LL[:,:,:,np.newaxis] * background.d1_scaling_matrix
    # To deal with the scaling matrix indices being jki in stead of ijk 
    a_times_d1_s = np.moveaxis(a_times_d1_s, 3, 1)   # xbca -> xabc
    s_times_d1_a = np.moveaxis(s_times_d1_a, 3, 1)   # xbca -> xabc

    # Laplacian & Hessian of lapse 
    D2_lapse = em4phi * (np.einsum('xij,xij->x', bar_gamma_UU, d2.lapse)
                         - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.lapse)
                         + 2.0*np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.lapse, d1.phi))

    # A^L_L contractions & DkDl of lapse
    bar_A_LL_A_UL = np.einsum('xaj,xkj,xal->xkl', bar_gamma_UU, bar_A_LL, bar_A_LL)

    DkDl_lapse = ( d2.lapse
                  - np.einsum('xkij,xk->xij', bar_chris, d1.lapse)
                  - 2.0*np.einsum('xi,xj->xij', d1.phi, d1.lapse)
                  - 2.0*np.einsum('xj,xi->xij', d1.phi, d1.lapse)) 
    
    DiDj_u = ( matter.d2_u 
                  - np.einsum("xmij, xm->xij",bar_chris, matter.d1_u)
                  - 2 * np.einsum("xi, xj->xij",matter.d1_u, d1.phi)
                  - 2 * np.einsum("xj, xi->xij",matter.d1_u,d1.phi)
                  + 2 * np.einsum("xij, xml, xl, xm->xij",bar_gamma_LL, bar_gamma_UU, d1.phi, matter.d1_u))
    

    D_L_A_LL = e4phi[:, np.newaxis, np.newaxis, np.newaxis]*(  a_times_d1_s
                                                             + s_times_d1_a
                                                             - np.einsum("xmij, xmk->xijk",bar_chris, bar_A_LL)
                                                             - np.einsum("xmik, xjm->xijk",bar_chris, bar_A_LL)
                                                             - 2*np.einsum("xj, xik->xijk",d1.phi, bar_A_LL)
                                                             - 2*np.einsum("xk, xji->xijk",d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xij, xml, xl, xmk->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xik, xml, xl, xjm->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL))

    # Omega_L and Omega_LL    
    Omega_L = (- 4 * d2Lambdadduu[:,np.newaxis] * matter.v[:,np.newaxis]* matter.d1_u
                   - 4 * d1Lambdadu[:,np.newaxis] * (matter.d1_v + np.einsum("xjb, xib,xj->xi",bar_gamma_UU, bar_A_LL,matter.d1_u)
                   + one_third * bssn_vars.K[:,np.newaxis] * matter.d1_u))
    # BUGFIX ChatGPT: d2Lambdadduu -> d1Lambdadu in the object above !!!
    
    Omega_U = np.einsum("xij, xi->xj",gamma_UU, Omega_L)
    
    Omega_LL = ( 4 * d1Lambdadu[:,np.newaxis, np.newaxis] * 
                (DiDj_u + e4phi[:,np.newaxis,np.newaxis] * matter.v[:,np.newaxis,np.newaxis] * (bar_A_LL + bar_gamma_LL * one_third * bssn_vars.K[:,np.newaxis,np.newaxis]))
                    +4 * d2Lambdadduu[:,np.newaxis, np.newaxis] * np.einsum('xi,xj->xij', matter.d1_u, matter.d1_u))

    Trace_Omega = get_trace(Omega_LL, gamma_UU)
    TF_Omega_LL = Omega_LL - one_third * gamma_LL * Trace_Omega[:, np.newaxis, np.newaxis]
    TF_Omega_UU = np.einsum('xia,xjb,xab->xij', gamma_UU, gamma_UU, TF_Omega_LL)

    # rho_GB, S_GB_L
    rho_GB = Trace_Omega * Trace_M - 2*np.einsum('xij,xia,xib,xab->x', M_LL, gamma_UU, gamma_UU, Omega_LL)
    
    S_GB_L = (Omega_L * Trace_M[:,np.newaxis] 
                  + 2*Trace_Omega[:,np.newaxis]*(N_L + one_third * d1.K)
                  - 2*(np.einsum("xjb, xib, xj->xi",gamma_UU, M_LL, Omega_L)
                       + np.einsum("xjb,xib,xj->xi",gamma_UU, Omega_LL, N_L)
                       + one_third * np.einsum("xjb, xib, xj->xi",gamma_UU, Omega_LL, d1.K)))

    # barred F, F_ij
    bar_F = ( ilapse * D2_lapse + Asquared - bssn_vars.K * bssn_vars.K )
    
    bar_F_LL = (ilapse[:,np.newaxis,np.newaxis] * DkDl_lapse
                    + e4phi[:,np.newaxis,np.newaxis] * (bar_A_LL_A_UL - two_thirds * (bssn_vars.K[:,np.newaxis,np.newaxis] 
                                                                                      - ilapse[:,np.newaxis,np.newaxis] * div_shift[:,np.newaxis,np.newaxis]) * bar_A_LL))
   
    # bar Trace Free S_GB
    bar_TF_S_GB_LL = ( - two_thirds * TF_Omega_LL*(bar_F[:,np.newaxis, np.newaxis] 
                                                   + 2 * (ilapse[:,np.newaxis, np.newaxis] * D2_lapse[:,np.newaxis, np.newaxis] 
                                                          - Asquared[:,np.newaxis, np.newaxis]))
                            - 2 * TF_M_LL*(  Trace_Omega[:,np.newaxis, np.newaxis] 
                                                - 4*d2Lambdadduu[:,np.newaxis, np.newaxis]* (-(matter.v[:,np.newaxis, np.newaxis])*(matter.v[:,np.newaxis, np.newaxis]) 
                                                                                             + np.einsum("xij, xi, xj->x",gamma_UU,matter.d1_u,matter.d1_u)[:,np.newaxis, np.newaxis]) 
                                                - 4 * matter.dVdu(matter.u)[:,np.newaxis, np.newaxis] * d1Lambdadu[:,np.newaxis, np.newaxis])
                            -  two_thirds * Trace_Omega[:,np.newaxis, np.newaxis]*(bar_F_LL - one_third * gamma_LL*(ilapse[:,np.newaxis, np.newaxis] * D2_lapse[:,np.newaxis, np.newaxis] 
                                                                                                                    - Asquared[:,np.newaxis, np.newaxis]))
                            +  2 * (  np.einsum("xi, xj->xij",N_L, Omega_L) 
                                + one_third * np.einsum("xi, xj->xij",d1.K, Omega_L) 
                                + np.einsum("xi, xj->xij",Omega_L, N_L) 
                                + one_third * np.einsum("xi, xj-> xij",Omega_L, d1.K))
                            +  2 * (  np.einsum("xik, xck, xjc->xij",TF_Omega_LL, gamma_UU, bar_F_LL) 
                                + np.einsum("xjk, xck, xic->xij",TF_Omega_LL, gamma_UU, bar_F_LL)
                                - 2 * np.einsum("xk, xck, xcij-> xij",Omega_L, gamma_UU, D_L_A_LL)
                                + np.einsum("xk, xck, xjic-> xij",Omega_L, gamma_UU, D_L_A_LL)
                                + np.einsum("xk, xck, xijc-> xij", Omega_L, gamma_UU, D_L_A_LL))
                            -   four_thirds * (  np.einsum("xij, xkl, xkl->xij",gamma_LL, TF_Omega_UU, bar_F_LL)
                                            + 2 * np.einsum("xij, xk, xk->xij", gamma_LL, Omega_U, N_L)
                                            + np.einsum("xij, xk, xk->xij",gamma_LL, Omega_U, d1.K))
                            -  8 * (d1Lambdadu[:,np.newaxis, np.newaxis] * d1Lambdadu[:,np.newaxis, np.newaxis] * TF_M_LL * bar_L_GB[:,np.newaxis, np.newaxis])) 
    
    # bar_S_GB (scalar)  
    bar_S_GB = (four_thirds * Trace_Omega * bar_F
                + 4 * Trace_M * (- d2Lambdadduu* (- (matter.v)*(matter.v) + (np.einsum("xij, xi, xj->x",gamma_UU, matter.d1_u, matter.d1_u)))
                                    - d1Lambdadu * matter.dVdu(matter.u) + one_third * Trace_Omega)
                - rho_GB
                - 2 * (np.einsum("xij, xij->x",TF_Omega_UU, M_LL) + np.einsum("xij, xij->x", TF_Omega_UU, bar_F_LL))
                - 4 * np.einsum("xij, xj, xi->x",gamma_UU, N_L, Omega_L)
                + 4 * d1Lambdadu * d1Lambdadu * Trace_M * bar_L_GB)
    
    # Store
    gb_vars.rho_GB[:]                = rho_GB
    gb_vars.S_GB_L[:]                = S_GB_L
    gb_vars.bar_TraceFree_S_GB_LL[:] = bar_TF_S_GB_LL
    gb_vars.bar_S_GB[:]              = bar_S_GB
    gb_vars.Omega_LL[:]              = Omega_LL

