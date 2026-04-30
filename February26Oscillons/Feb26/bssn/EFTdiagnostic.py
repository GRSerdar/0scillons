import numpy as np

from core.grid import *
from bssn.tensoralgebra import *
from bssn.bssnvars import BSSNVars
from bssn.ModifiedGravity import GBVars, get_gb_core, get_esgb_br_terms
from bssn.bssnrhs_MG import get_bssn_rhs

_EPS = 1e-30


def _sigmoid_switch(chi, chi0):
    return 1.0 / (1.0 + np.exp(-100.0 * (chi - chi0)))


def get_eft_diagnostic(states_over_time, t, grid, background, matter, params, log_progress=True):
    """
    EFT weak-coupling diagnostic.
    Three terms :
      1) Term 1: |grad u| = sqrt(max(G, 0)) with G = -v^2 + gamma^{ij} d_i u d_j u 
      2) nabla_mu nabla_nu \phi \;I_2^{1/4} where I_2 = -S_nn^2 + 2 gamma^{ij} C_i C_j + C^{ij} C_{ij}
      3) |L_GB|^{1/4} from full_L_GB AFTER the EsGB matrix solve.
    """
    lambda_GB, a_mg, b_mg, chi0, coupling, g2 = params

    r = grid.r
    N = grid.num_points
    num_times = len(t) if np.ndim(t) > 0 else 1

    # Constructing objects that will be used
    term_grad_u = np.zeros((num_times, N))
    hess_I2 = np.zeros((num_times, N))
    term_hess_u_sqrt = np.zeros((num_times, N))
    term_gb_quarter = np.zeros((num_times, N))
    L_inv = np.zeros((num_times, N))
    L = np.zeros((num_times, N))
    dominant_term_idx = np.zeros((num_times, N), dtype=np.int8)

    # Only bare couplings are tracked; GRChombo operates on bare f'(phi) directly, so
    # the code-level (sigmoid/Sigma-dressed) versions are not comparable and are disabled.
    # lambda_prime_code = np.zeros((num_times, N))
    lambda_prime_bare = np.zeros((num_times, N))
    # g2_code = np.zeros((num_times, N))
    g2_bare = np.zeros((num_times, N))

    # wc_lambda_code = np.zeros((num_times, N))
    wc_lambda_bare = np.zeros((num_times, N))
    # wc_g2_code = np.zeros((num_times, N))
    wc_g2_bare = np.zeros((num_times, N))

    # --- g2-specific length scale L_tilde^{-1} = max(|K_phi|, |D_i phi D^i phi|^{1/2}) -----
    # Used only for the g2 weak-coupling condition; lambda'/GB still uses the full L^{-1}.
    # Convention note: |K_phi| is sign-independent; with v = +n^mu d_mu u we simply use |v|.
    K_phi_arr             = np.zeros((num_times, N))
    grad_phi_spatial_arr  = np.zeros((num_times, N))
    L_tilde_inv_arr       = np.zeros((num_times, N))
    L_tilde_arr           = np.zeros((num_times, N))
    dominant_g2_term_idx  = np.zeros((num_times, N), dtype=np.int8)

    # --- Hyperbolicity diagnostics -------------------------------------------
    # GB channel: det_M, disc_GB = det_M / (1 + Omega_pp)^2, plus Omega_pp itself.
    # g2 channel: disc_g2 = factor1^3 * factor2 with factor1, factor2 recorded too.
    # ok-masks flag grid points where the respective hyperbolicity condition holds.
    det_M_arr    = np.zeros((num_times, N))
    disc_GB_arr  = np.zeros((num_times, N))
    denominator_GB_arr  = np.zeros((num_times, N))
    Omega_pp_arr = np.zeros((num_times, N))
    Omega_UU_arr     = np.zeros((num_times, N, 3, 3))
    Omega_perp_U_arr = np.zeros((num_times, N, 3))
    disc_g2_arr  = np.zeros((num_times, N))
    factor1_arr  = np.zeros((num_times, N))
    factor2_arr  = np.zeros((num_times, N))
    hyp_GB_ok    = np.zeros((num_times, N), dtype=bool)
    hyp_g2_ok    = np.zeros((num_times, N), dtype=bool)

    for i in range(num_times):
        state = states_over_time if num_times == 1 else states_over_time[i]
        state = state.reshape(grid.NUM_VARS, -1)

        bssn = BSSNVars(N)
        bssn.set_bssn_vars(state)
        matter.set_matter_vars(state, bssn, grid)

        d1 = grid.get_d1_metric_quantities(state)
        d2 = grid.get_d2_metric_quantities(state)

        em4phi = np.exp(-4.0 * bssn.phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn.h_LL, background)
        bar_gamma_UU = get_bar_gamma_UU(r, bssn.h_LL, background)
        gamma_UU = em4phi[:, np.newaxis, np.newaxis] * bar_gamma_UU
        
        ############# Modified Gravity Terms ################
        
        gb = GBVars(N)
        get_gb_core(gb, r, bssn, d1, d2, grid, background, lambda_GB, chi0)
        get_esgb_br_terms(gb, r, matter, bssn, d1, d2, grid, background, lambda_GB, chi0, coupling, g2)
        bssn_rhs_tmp = BSSNVars(N)
        emtensor = matter.get_emtensor(r, bssn, background, gb)
        _dudt, dPidt = get_bssn_rhs(
            bssn_rhs_tmp, r, matter, bssn, d1, d2, grid, background, gb, (a_mg, b_mg), emtensor)

        _, Delta_ULL, _ = get_tensor_connections(r, bssn.h_LL, d1.h_LL, background)
        bar_chris = get_bar_christoffel(r, Delta_ULL, background)

        #### Length Scale for GB terms #####
        ############# Term 1 ################
        grad_phi_sq = np.einsum("xij,xi,xj->x", gamma_UU, matter.d1_u, matter.d1_u)
        G_kinetic = -matter.v * matter.v + grad_phi_sq

        ############# Term 2 ################
        """
        In this term we have to perform different projections of nabla_\mu nabla_\nu \phi
        (1) spatial-spatial : C_ij
        (2) normal-spatial  : C_i
        (3) normal-normal   : C_nn
        """
        chi_c = np.maximum(em4phi, 1e-6)
        d1_chi = -4.0 * em4phi[:, np.newaxis] * d1.phi

        # This is \bar{D}_i \bar{D}_j u
        bar_DiDju = matter.d2_u - np.einsum("xmij,xm->xij", bar_chris, matter.d1_u)

        # We want to convert this to DiDj u
        du_dchi = np.einsum("xij,xi,xj->x", bar_gamma_UU, matter.d1_u, d1_chi)
        
        DiDju = bar_DiDju + (0.5/ chi_c[:, np.newaxis, np.newaxis] 
                            * (matter.d1_u[:, :, np.newaxis] * d1_chi[:, np.newaxis, :]
                                + d1_chi[:, :, np.newaxis] * matter.d1_u[:, np.newaxis, :]
                                - bar_gamma_LL * du_dchi[:, np.newaxis, np.newaxis]))

        bar_A_LL = get_bar_A_LL(r, bssn, background)
        K_LL = bar_A_LL + one_third * bar_gamma_LL * bssn.K[:, np.newaxis, np.newaxis]
        
        # (1) Spatial-Spatial
        Cij = DiDju - K_LL * (matter.v / chi_c)[:, np.newaxis, np.newaxis] 
        
        # (2) Normal-Spatial
        Ci = (matter.d1_v - (bssn.K / 3.0)[:, np.newaxis] * matter.d1_u 
                - np.einsum("xjk,xij,xk->xi", bar_gamma_UU, bar_A_LL, matter.d1_u))

        # (3) Normal-Normal 
        # IS Advection really neede here ? Normally yes in Engrenage, but lets be carefull.
        advec_v_dot = np.einsum("xj,xj->x", background.inverse_scaling_vector * bssn.shift_U, matter.advec_v)
        # SIGN CHANGE, CHANGED THIS TO + TO KEEP CONVENTION OF ENGRENAGE-GRCHOMBO; K_\phi = - \Pi
        Cnn_times_lapse = dPidt + 0*advec_v_dot - em4phi * np.einsum("xij,xi,xj->x", bar_gamma_UU, matter.d1_u, d1.lapse)
        lapse_s = np.maximum(bssn.lapse, 1e-10)
        S_nn = Cnn_times_lapse / lapse_s

        # Puttin the three components together.
        # Raised with wrong non physical metric

        # Ci_sq = np.einsum("xij,xi,xj->x", bar_gamma_UU, Ci, Ci)
        # Cij_sq = np.einsum("xia,xjb,xij,xab->x", bar_gamma_UU, bar_gamma_UU, Cij, Cij)
        
        #correct physical metric raising
        Ci_sq = np.einsum("xij,xi,xj->x", gamma_UU, Ci, Ci)
        Cij_sq = np.einsum("xia,xjb,xij,xab->x", gamma_UU, gamma_UU, Cij, Cij)
        
        I2 = -S_nn * S_nn + 2.0 * Ci_sq + Cij_sq

        ############# Term 3 ################
        gb_scalar = gb.full_L_GB

        # takking the correct power of the different terms.
        t_grad = np.sqrt(np.maximum(G_kinetic, 0.0))
        t_hess = np.power(np.maximum(I2, 0.0), 0.25)
        t_gb = np.power(np.maximum(np.abs(gb_scalar), 0.0), 0.25)

        # making new array with extra dimension
        terms = np.stack([t_grad, t_hess, t_gb], axis=0)
        # Calculate inverse length scale
        l_inv = np.max(terms, axis=0)
        # Which physical term "dom"inates
        dom = np.argmax(terms, axis=0).astype(np.int8)

        # Bare couplings only (match GRChombo): strip the sigmoid/Sigma damping that
        # is baked into gb.d1Lambdadu and gb.Sigma at the code level.
        chi = em4phi
        S = _sigmoid_switch(chi, chi0)
        S_ell = np.maximum(S * np.abs(lambda_GB), _EPS)

        # LAMBDA COUPLING Scales
        # We take the absolute value of the first derivative of the coupling function 
        # since this appears in the EOM of the scalar field.
        _lam_code_internal = np.abs(gb.d1Lambdadu)  # only used to recover the bare value
        # Bare first derivative of coupling
        lam_bare = _lam_code_internal / S_ell if lambda_GB != 0 else np.zeros(N)
        # Length scale
        L_here = 1.0 / np.maximum(l_inv, _EPS)


        ############# g2-specific length scale ################
        # This is a length scale calculated from the terms relevant for the g2 term

        K_phi = np.abs(matter.v) # normally K_phi = -\Pi but due to the absolute value here it makes no difference
        grad_phi_spatial = np.sqrt(np.maximum(grad_phi_sq, 0.0))
        L_tilde_inv = np.maximum(K_phi, grad_phi_spatial)
        L_tilde = 1.0 / np.maximum(L_tilde_inv, _EPS)
        terms_g2 = np.stack([K_phi, grad_phi_spatial], axis=0)
        dom_g2 = np.argmax(terms_g2, axis=0).astype(np.int8)
        # g2 COUPLING
        g2_eff_bare = np.full(N, np.abs(g2))


        ############# Weak Coupling Conditions: #############
        # LAMBDA (GB) channel uses the full three-term L^{-1}
        wc_lam_bare = np.sqrt(np.maximum(lam_bare, 0.0)) * l_inv
        # g2 channel uses the g2-specific length scale L_tilde^{-1}
        wc_g2_b = np.sqrt(np.maximum(g2_eff_bare, 0.0)) * L_tilde_inv

        # Storing results
        term_grad_u[i] = t_grad
        hess_I2[i] = I2
        term_hess_u_sqrt[i] = t_hess
        term_gb_quarter[i] = t_gb
        L_inv[i] = l_inv
        L[i] = L_here
        dominant_term_idx[i] = dom

        lambda_prime_bare[i] = lam_bare
        g2_bare[i] = g2_eff_bare

        wc_lambda_bare[i] = wc_lam_bare
        wc_g2_bare[i] = wc_g2_b

        K_phi_arr[i]            = K_phi
        grad_phi_spatial_arr[i] = grad_phi_spatial
        L_tilde_inv_arr[i]      = L_tilde_inv
        L_tilde_arr[i]          = L_tilde
        dominant_g2_term_idx[i] = dom_g2

        ############# Hyperbolicity Diagnostics ################
        """
        In this diangostic we will check the hyperbolicity 
        two different diagnostics for different contributions to the principal symbol
        - GB contribution 
        - g2 contribution
        """
        ##########################
        # Gauss Bonnet Diagnostic
        ##########################

        # hyper_Cmunu = lambda''(phi) \partial_v\phi \partial_\nu\phi + \lambda'(\phi) [\nabla_\mu \nabla_\nu \phi], 
        # where the last part of the last term (\nabla_\mu \nabla_\nu \phi) corresponds with the I^2 object we calculated in the previous diagnostic
        
        #We take the couupling functions (not the absolute value since in this diagnostic the sign matters). 
        #We divide by the S function for completeness, but it won't make a difference.
        dfGB = gb.d1Lambdadu / S_ell if lambda_GB != 0 else np.zeros(N)
        df2GB = gb.d2Lambdadduu / S_ell if lambda_GB != 0 else np.zeros(N)

        # We define an extra prefactor to take into account the different conventions ...
        pref_GB = 4* 2* eight_pi_G   # = 64πG

        # (1) Spatial-spatial sector of nabla_mu nabla_nu lambda(phi)
        hyper_Cij = pref_GB* (dfGB[:, np.newaxis, np.newaxis] * Cij
                                + df2GB[:, np.newaxis, np.newaxis] 
                                * matter.d1_u[:, :, np.newaxis] * matter.d1_u[:, np.newaxis, :])

        # (2) Normal-spatial sector
        hyper_Ci = -pref_GB* (dfGB[:, np.newaxis] * Ci
                                + df2GB[:, np.newaxis] * matter.v[:, np.newaxis] * matter.d1_u)

        # (3) Normal-normal sector
        hyper_C = pref_GB*(dfGB * S_nn + df2GB * matter.v * matter.v)

        # Now the Omega Objects which appear in PhD Llibert p.62 can be implemented
        Omega_UU = chi_c[:, np.newaxis, np.newaxis]* chi_c[:, np.newaxis, np.newaxis] * np.einsum("xia,xjb,xab->xij", bar_gamma_UU, bar_gamma_UU, hyper_Cij)
        Omega_perp_U = chi_c[:, np.newaxis]*np.einsum("xij,xj->xi", bar_gamma_UU, hyper_Ci)
        Omega_pp = hyper_C  

        # Now we define the ratio of determinants 
        # Here we define the upper metric again, but with the regularized chi
        g_UU = chi_c[:, np.newaxis, np.newaxis] * bar_gamma_UU  
        #beta_U = bssn.shift_U  (non scaled shift)
        beta_U = background.inverse_scaling_vector * bssn.shift_U
                                 
        alpha_s = lapse_s 

        M_eff = (1.0 / chi_c[:, np.newaxis, np.newaxis]) * ((g_UU - Omega_UU) * (1.0 + Omega_pp[:, np.newaxis, np.newaxis])
                    - (1.0 / alpha_s[:, np.newaxis, np.newaxis]) * (Omega_perp_U[:, :, np.newaxis] * beta_U[:, np.newaxis, :]
                                                                        + Omega_perp_U[:, np.newaxis, :] * beta_U[:, :, np.newaxis]) 
                    - Omega_pp[:, np.newaxis, np.newaxis] * (1/ alpha_s[:, np.newaxis, np.newaxis]**2) *(beta_U[:, :, np.newaxis] * beta_U[:, np.newaxis, :]) 
                    + Omega_perp_U[:, :, np.newaxis] * Omega_perp_U[:, np.newaxis, :])

        # This object is the normalised determinant, which typically has values of order one and is normalised to unity in the absence of a scalar field.
        det_M_i = np.linalg.det(M_eff)

        # Full ratio with correct factor in front 
        disc_GB_i = det_M_i # / np.maximum((1.0 + Omega_pp)**2, _EPS)
        denominator_GB = np.maximum((1.0 + Omega_pp)**2, _EPS)
        ##########################
        # g2 Diagnostic
        ##########################
        
        # We can rewrite the det(-P^{\mu\nu}_mm) with the matrix determinatn Lemma
        # which simplifies it a lot

        # det(-(94))/det(g^mu nu), where Equation 94 from https://arxiv.org/abs/2101.11623
        factor1 = 1.0 - g2 * G_kinetic          # (1 + 2g2*X) with X = -G/2
        factor2 = 1.0 - 3.0 * g2 * G_kinetic    # (1 + 2g2*X - 2g2*G)

        disc_g2_i = factor1**3 * factor2

        # Store hyperbolicity diagnostics for this time slice
        det_M_arr[i]    = det_M_i
        disc_GB_arr[i]  = disc_GB_i
        denominator_GB_arr[i] = denominator_GB
        Omega_pp_arr[i] = Omega_pp
        Omega_UU_arr[i]     = Omega_UU
        Omega_perp_U_arr[i] = Omega_perp_U

        disc_g2_arr[i]  = disc_g2_i
        factor1_arr[i]  = factor1
        factor2_arr[i]  = factor2
        # Strong-hyperbolicity-style masks: positive discriminant (and positive
        # factor1 for g2, since factor1^3 preserves sign so disc_g2>0 alone is
        # insufficient).
        hyp_GB_ok[i] = disc_GB_i > 0.0
        hyp_g2_ok[i] = (factor1 > 0.0) & (factor2 > 0.0)


        # Same boundary treatment as other grid quantities in the code.
        for arr in (
            term_grad_u[i], hess_I2[i], term_hess_u_sqrt[i], term_gb_quarter[i], L_inv[i], L[i],
            lambda_prime_bare[i], g2_bare[i],
            wc_lambda_bare[i], wc_g2_bare[i],
            K_phi_arr[i], grad_phi_spatial_arr[i], L_tilde_inv_arr[i], L_tilde_arr[i],
            det_M_arr[i], disc_GB_arr[i], Omega_pp_arr[i],
            disc_g2_arr[i], factor1_arr[i], factor2_arr[i],
        ):
            grid.fill_inner_boundary_single_variable(arr)

    return {
        "t": t,
        "r": r,
        "term_grad_u": term_grad_u,
        "hess_I2": hess_I2,
        "term_hess_u_sqrt": term_hess_u_sqrt,
        "term_gb_quarter": term_gb_quarter,
        "L_inv": L_inv,
        "L": L,
        "dominant_term_idx": dominant_term_idx,
        "dominant_grad_mask": dominant_term_idx == 0,
        "dominant_hess_mask": dominant_term_idx == 1,
        "dominant_gb_mask": dominant_term_idx == 2,
        "lambda_prime_bare": lambda_prime_bare,
        "g2_bare": g2_bare,
        "wc_lambda_bare": wc_lambda_bare,
        "wc_g2_bare": wc_g2_bare,
        "wc_lambda_bare_ok": wc_lambda_bare < 1.0,
        "wc_g2_bare_ok": wc_g2_bare < 1.0,
        # g2-specific length scale (used by wc_g2_bare above)
        "K_phi": K_phi_arr,
        "grad_phi_spatial": grad_phi_spatial_arr,
        "L_tilde_inv": L_tilde_inv_arr,
        "L_tilde": L_tilde_arr,
        "dominant_g2_term_idx": dominant_g2_term_idx,
        "dominant_g2_Kphi_mask": dominant_g2_term_idx == 0,
        "dominant_g2_gradphi_mask": dominant_g2_term_idx == 1,
        # Hyperbolicity diagnostics
        "det_M": det_M_arr,
        "disc_GB": disc_GB_arr,
        "Omega_pp": Omega_pp_arr,
        "Omega_UU":     Omega_UU_arr,
        "Omega_perp_U": Omega_perp_U_arr,
        "disc_g2": disc_g2_arr,
        "factor1_g2": factor1_arr,
        "factor2_g2": factor2_arr,
        "hyp_GB_ok": hyp_GB_ok,
        "hyp_g2_ok": hyp_g2_ok,
    }
