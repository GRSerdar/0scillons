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

      2) I_2^{1/4} where I_2 = -S_nn^2 + 2 gamma^{ij} C_i C_j + C^{ij} C_{ij}
         with C_ij, C_i from GRChombo-style spatial Hessian assembly and
         S_nn = Cnn_times_lapse / alpha, Cnn_times_lapse = -rhs_v + advec_v
         - chi * (gamma^{ij} d_i u d_j alpha).

      3) |L_GB|^{1/4} from full_L_GB after the EsGB matrix solve.

    L^{-1} = max of those three, so L is the shortest dynamical length scale.
    Weak-coupling check: sqrt(|coupling|) / L == sqrt(|coupling|) * L^{-1} << 1.
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

        ############# Term 1 ################
        G_kinetic = -matter.v * matter.v + np.einsum("xij,xi,xj->x", gamma_UU, matter.d1_u, matter.d1_u)

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
        Cnn_times_lapse = -dPidt + 0*advec_v_dot - em4phi * np.einsum("xij,xi,xj->x", bar_gamma_UU, matter.d1_u, d1.lapse)
        lapse_s = np.maximum(bssn.lapse, 1e-10)
        S_nn = Cnn_times_lapse / lapse_s

        # Puttin the three components together.
        Ci_sq = np.einsum("xij,xi,xj->x", bar_gamma_UU, Ci, Ci)
        Cij_sq = np.einsum("xia,xjb,xij,xab->x", bar_gamma_UU, bar_gamma_UU, Cij, Cij)
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

        # LAMBDA COUPLING
        _lam_code_internal = np.abs(gb.d1Lambdadu)  # only used to recover the bare value
        lam_bare = _lam_code_internal / S_ell if lambda_GB != 0 else np.zeros(N)

        # g2 COUPLING
        g2_eff_bare = np.full(N, np.abs(g2))

        # Length scale
        L_here = 1.0 / np.maximum(l_inv, _EPS)

        # Weak Coupling Conditions:
        #   sqrt(|coupling|) / L  ==  sqrt(|coupling|) * L^{-1}  (much less than 1 for EFT validity)
        # LAMBDA
        wc_lam_bare = np.sqrt(np.maximum(lam_bare, 0.0)) * l_inv
        # g2
        wc_g2_b = np.sqrt(np.maximum(g2_eff_bare, 0.0)) * l_inv

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

        # Same boundary treatment as other grid quantities in the code.
        for arr in (
            term_grad_u[i], hess_I2[i], term_hess_u_sqrt[i], term_gb_quarter[i], L_inv[i], L[i],
            lambda_prime_bare[i], g2_bare[i],
            wc_lambda_bare[i], wc_g2_bare[i],
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
    }
