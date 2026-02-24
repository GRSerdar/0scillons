# constraintsolver

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# import modified gravity objects
from bssn.ModifiedGravity import GBVars, get_gb_core, get_esgb_br_terms

from core.grid import *
from bssn.tensoralgebra import *

class CTTKBHConstraintSolver :
    """Solves the constraints for a BH plus scalar configuration."""

    def __init__(self, grid, a_MBH, a_scalar_mass, parameters=(0.05, 0.2, 0.4, 0.15, 'quadratic')) :

        # Extra variables defined to include modified gravity terms
        a_r = grid.r
        N = grid.N

        # Checks set to false
        self.background_set = False
        self.matter_source_set = False

        # Set params
        self.scalar_mass = a_scalar_mass
        self.MBH = a_MBH        
        
        # to be set later or elsewhere
        self.u = []
        self.v = []
        self.dudr = []
        self.psi = []
        self.grr = []
        self.gtt = []
        self.gpp = []
        self.Arr = []
        self.Att = []
        self.App = []
        self.AijAij =[]
        self.Lap_psi_over_psi5 = []
        
        
        # Modified gravity objects
        self.gb_vars = GBVars(N)
        self.lambda_GB, self.a, self.b, self.chi0, self.coupling = parameters

        # Set up the solver uniform grid
        self.r = a_r
        Rmax_approx = a_r[-1]+0.1 # set this roughly, fixed below
        dr = a_r[3]/2.0
        Rmin = self.MBH / 10.0 + dr/2.0
        #print(dr, Rmin, Rmax_approx)

        # Work out R vector
        num_points = int((Rmax_approx - Rmin)/dr) + 1
        Rmax = Rmin + (num_points - 1) * dr
        self.R = np.linspace(Rmin, Rmax, num_points)

        # this is bar{gamma}_ij = psi^{-4} gamma_ij, flat conformal metric
        self.hrr = 1.0  
        self.htt = self.R * self.R
        self.hpp = self.R * self.R #sintheta = 1        
        
        # Set background solution
        self.set_BH_background_solution()
        
        # Set remaining vars to their initial values
        self.K = self.K0
        self.Wr = self.Wr0
        self.dWrdr = self.dWrdr0
        self.Q = self.Q0
        self.update_Aij_vars()
        self.AijAij0 = self.AijAij
        self.Arr0 = self.Arr
        self.Att0 = self.Att
        self.App0 = self.App
        
    
    def get_evolution_vars(self) :
        
        assert (self.background_set and self.matter_source_set), "BG and source not set"
        
        # Solve for the constraint vars
        error = 10.0
        tol = 1.0e-2
        max_iter = 50
        iteration = 0
        while ((error > tol) and (iteration < max_iter)):
            # Because the convergence is oscillatory, best to only add a fraction of the correction each time
            frac = (iteration + 0.5 * max_iter) / max_iter
            frac = min(frac, 1.0)
            deltaK = self.get_deltaK() # Because Aij has changed
            self.K = self.K0 + deltaK
            Mom = self.get_Mom()
            cs = CubicSpline(self.R, deltaK)
            ddeltaKdr = cs(self.R, 1) # first derivative of dK
            # Solve for correction to Wr, sourced by change in dKdr
            Wr, dWrdr, Q = self.solve_for_Wr(ddeltaKdr)
            self.Wr = self.Wr0 + frac * Wr
            self.dWrdr = self.dWrdr0 + frac * dWrdr
            self.Q = self.Q0 + frac * Q
            self.update_Aij_vars() # Need full Aijs for the Ham constraint
            Ham = self.get_Ham()
            error = np.linalg.norm(Ham) + np.linalg.norm(Mom)
            iteration = iteration + 1
            #print("error is ", error, " after ", iteration)
            color = cm.viridis(iteration / max_iter)   # Nmax = total iterations you expect
            Ham = self.get_Ham()
            Mom = self.get_Mom()
            #plt.plot(self.R, Ham, '-',color=color, label=f"{iteration}")
            #plt.plot(self.R, Mom, '--',color=color, label=f"{iteration}")
            #plt.legend(loc='best')
            #plt.grid()
        
        print("error is: ", error, " after iter: ", iteration)
 
        # One final iteration to make sure we add the full correction at least once
        deltaK = self.get_deltaK() # Because Aij has changed
        self.K = self.K0 + deltaK
        cs = CubicSpline(self.R, deltaK)
        ddeltaKdr = cs(self.R, 1) # first derivative of dK
        # Solve for correction to Wr, sourced by change in dKdr
        Wr, dWrdr, Q = self.solve_for_Wr(ddeltaKdr)
        self.Wr = self.Wr0 + frac * Wr
        self.dWrdr = self.dWrdr0 + frac * dWrdr        

        # Fix Ham constraint last as we can do this to machine precision
        deltaK = self.get_deltaK()
        self.K = self.K0 + deltaK 

        # Plot to check internal measure of constraints
        """
        Ham = self.get_Ham()
        Mom = self.get_Mom()
        plt.plot(self.R, Ham, '-', label="Ham")
        plt.plot(self.R, Mom, '--',label = "Mom")
        plt.legend(loc='best')
        plt.ylim(-1.0, 1.0)
        plt.grid()
        """
        
        # Convert quantities into the evolution vars
        psi4_r = self.psi **4.0
        K_r = self.K
        arr_r = self.Arr * self.psi ** (-6.0)
        att_r = self.Att * self.psi ** (-6.0) / self.R / self.R
        
        # Interpolate back onto the grid r
        cs1 = CubicSpline(self.R, psi4_r)
        psi4 = cs1(self.r)
        cs2 = CubicSpline(self.R, K_r)
        K = cs2(self.r)        
        cs3 = CubicSpline(self.R, arr_r)
        arr = cs3(self.r) 
        cs4 = CubicSpline(self.R, att_r)
        att = cs4(self.r)
        app = att
        
        return psi4, K, arr, att, app
    
    # Set the matter vars, which will give the source
    #def set_matter_source(self, a_u, a_v, a_dudr) :
    def set_matter_source(self, a_u, a_v, a_dudr, d1 , d2 , scalar_matter, bssn_vars, background, grid):

        # Interpolate these sources onto R
        cs1 = CubicSpline(self.r, a_u)
        self.u = cs1(self.R)
        
        cs2 = CubicSpline(self.r, a_v)
        self.v = cs2(self.R)        

        cs2 = CubicSpline(self.r, a_dudr)
        self.dudr = cs2(self.R)

        #################################################################################
        # Modified Gravity Changes

        get_gb_core(self.gb_vars, grid.r , bssn_vars, d1, d2, grid, background,
                lambda_GB=self.lambda_GB, chi0=self.chi0)
        
        get_esgb_br_terms(self.gb_vars, grid.r , scalar_matter, bssn_vars, d1, d2, grid,
                      background, lambda_GB=self.lambda_GB, chi0=self.chi0, coupling=self.coupling)

        # Calculate the necessary MG terms:
        rho_GB_r = self.gb_vars.rho_GB                     
        S_GB_L = self.gb_vars.S_GB_L                      
        S_GB_r = S_GB_L[:,0] # Only radial direction due to spherical symmetry

        # We interpolate them on the constraint solver grid
        cs_rhoGB = CubicSpline(self.r, rho_GB_r)
        rho_GB   = cs_rhoGB(self.R)

        cs_SGB   = CubicSpline(self.r, S_GB_r)
        S_GB   = cs_SGB(self.R)
        #################################################################################

        # Rho (adding modified gravity terms)
        self.rho = (0.5 * self.scalar_mass * self.scalar_mass * self.u * self.u 
                    + 0.5 / (self.psi**4.0) * self.dudr * self.dudr
                    + 0.5 * self.v * self.v)
        
        self.rho  += rho_GB
        
        # S_L (adding modified gravity terms)
        self.SiU = self.v * self.dudr

        self.SiU  += S_GB
        
        # Adjust psi solution for Schwazschild de Sitter??
        #CC = eight_pi_G * self.rho
        #self.psi = np.sqrt(1.0 + self.MBH/self.R - CC * self.R * self.R)        
        #self.Lap_psi_over_psi5 = -0.25 * ((self.MBH**2.0 + 16.0 * CC * self.MBH * self.R**3.0 
        #                                  + 4.0 * CC * self.R**4.0 * (3.0 - 2.0 * CC * self.R**2.0)) 
        #                                  / ((self.MBH + self.R - CC * self.R**3.0)**4.0))
        """
        # We don't really need this
        if self.MBH == 0.0:
            H0  = np.sqrt(np.mean(self.rho) / 3.0)    # We are working in 8piG = 1, but if not, that factor could be implemented here i think.
            print("H0: ", H0)
            K_FLRW  = -3.0 * H0                                 
            self.K0 = K_FLRW * np.ones_like(self.R) 
        """
        self.matter_source_set = True        
        
    # Set BH BG solution
    def set_BH_background_solution(self) :
        
        self.psi = (1.0 + 0.5 * self.MBH/self.R)
        self.grr = self.psi**4.0 * self.hrr
        self.gtt = self.psi**4.0 * self.htt
        self.gpp = self.psi**4.0 * self.hpp
        self.Lap_psi_over_psi5 = np.zeros_like(self.R)
        
        self.K0 = np.zeros_like(self.R)

        self.Wr0 = np.zeros_like(self.R)
        self.dWrdr0 = np.zeros_like(self.R)
        self.Q0 = self.dWrdr0 + 2.0 * self.Wr0 / self.R  
        
        # Thomas solution
        #self.psi = np.sqrt(1.0 + self.MBH/self.R)
        #self.grr = self.psi**4.0 * self.hrr
        #self.gtt = self.psi**4.0 * self.htt
        #self.gpp = self.psi**4.0 * self.hpp
        #self.Lap_psi_over_psi5 = - 0.25 * self.MBH * self.MBH * (self.MBH + self.R)**(-4.0)
        #self.K0 = - self.MBH / (self.R + self.MBH) / (self.R + self.MBH)
        #R2 = self.R * self.R
        #self.Wr0 = - (0.5 * self.MBH / self.R + self.MBH * self.MBH / 3.0 / R2)
        #self.dWrdr0 = (0.5 * self.MBH / R2 + 2.0 / 3.0 * self.MBH * self.MBH / R2 / self.R)
        #self.Q0 = self.dWrdr0 + 2.0 * self.Wr0 / self.R      
        
        self.background_set = True

    # Extra funciton in the case of reading in tabulated stable oscillon ID
    def set_oscillaton_background(self, psi_on_R):
        """
        We use a given conformal factor psi_on_R as the background instead of the BH background.
        """

        # Set psi and the corresponding metric components
        self.psi = psi_on_R
        self.grr = self.psi**4.0 * self.hrr
        self.gtt = self.psi**4.0 * self.htt
        self.gpp = self.psi**4.0 * self.hpp

        # Compute Lap(psi)/psi^5 in flat-space spherical coordinates
        # Lap psi = d2psi/dr2 + (2/r) dpsi/dr
        cs_psi    = CubicSpline(self.R, self.psi)
        dpsi_dr   = cs_psi(self.R, 1)
        d2psi_dr2 = cs_psi(self.R, 2)

        Lap_psi = d2psi_dr2 + 2.0/self.R * dpsi_dr
        self.Lap_psi_over_psi5 = Lap_psi / (self.psi**5.0)

        self.Lap_psi_over_psi5 = Lap_psi / (self.psi**5.0)

        # Same as bh example
        self.K0     = np.zeros_like(self.R)   
        self.Wr0    = np.zeros_like(self.R)
        self.dWrdr0 = np.zeros_like(self.R)
        self.Q0     = self.dWrdr0 + 2.0 * self.Wr0 / self.R

        self.is_oscillaton   = True
        
        self.background_set = True

        
    def dydr_for_Wr(self, r_here, y, dKdr, psi) :
        """Returns the gradient dy/dr for the Mom Constraint"""
    
        Wr = y[0] 
        Qr = y[1] # Qr = dWdr + 2W/r 
    
        dydr = np.zeros_like(y)
        
        # This is dWrdr = Q - 2W/r 
        dydr[0] = Qr - 2 * Wr / r_here
        # This is dQdr, where Q = dWdr + 2W/r 
        
        #SiU = 0.0 (old version of the code)
        SiU = np.interp(r_here, self.R, self.SiU)

        dydr[1] = 0.75 * eight_pi_G * psi**10.0 * SiU + 0.5 * psi**6.0 * dKdr
    
        return dydr
    
    def integrate_using_midpoint(self, a_dydr, y0, a_dKdr) :
    
        y_solution = np.zeros([2,np.size(self.R)])
    
        a_dr = self.R[1] - self.R[0] # assume fixed
    
        for ir, r_here in enumerate(self.R) :
            if ir == 0 :
                y_solution[:,ir] = y0
                old_r = r_here
            else :
                dydr_at_r = a_dydr(old_r, y_solution[:,ir-1], a_dKdr[ir-1], self.psi[ir-1])
                
                y_r_plus_half = (y_solution[:,ir-1] + 0.5 *
                                          dydr_at_r * a_dr)
                
                r_plus_half = old_r + 0.5*a_dr
            
                dKdr_r_plus_half = 0.5 * (a_dKdr[ir-1] + a_dKdr[ir])
                psi_r_plus_half = 0.5 * (self.psi[ir-1] + self.psi[ir])
                
                dydr_at_r_plus_half = a_dydr(r_plus_half, y_r_plus_half, 
                                       dKdr_r_plus_half, psi_r_plus_half)
                
                y_solution[:,ir] = (y_solution[:,ir-1] + 
                                          dydr_at_r_plus_half * a_dr) 
                old_r = r_here
    
        Wr = y_solution[0, :]
        Qr = y_solution[1, :]
        dWrdr = Qr - 2.0/self.R * Wr
    
        return Wr, dWrdr, Qr
    
    def get_deltaK(self) :
        """Returns the value of K from the Ham constraint"""
    
        Ksquared = 12.0 * self.Lap_psi_over_psi5 + 1.5 * self.AijAij + 3.0 * eight_pi_G * self.rho
    
        minusK = - np.sqrt(np.abs(Ksquared))
    
        deltaK = minusK - self.K0
    
        return deltaK
    
    def solve_for_Wr(self, ddeltaKdr) :
    
        q0 = 0.0
        r0 = self.R[0]
        w0 = q0 * r0 / 3.0
        
        # test new value
        y0 = np.array([w0, q0])
        Wr_test, dWrdr_test, Q_test = self.integrate_using_midpoint(self.dydr_for_Wr, 
                                                               y0, ddeltaKdr)
    
        #print(q0, Q_test[-1])
    
        for i in np.arange(3) :
        # Correct the q0 values for the linear term
            q0 += - Q_test[-1]
            w0 = q0 * r0 / 3.0 
            y0 = np.array([w0, q0])
            Wr_test, dWrdr_test, Q_test = self.integrate_using_midpoint(self.dydr_for_Wr, 
                                                                        y0, ddeltaKdr)
            #print(q0, Q_test[-1])
    
        return Wr_test, dWrdr_test, Q_test
    
    def update_Aij_vars(self) :  
        
        R2 = self.R * self.R
        
        # These are \bar{A}_ij = psi^{2} A_ij (A^ij = psi^{-10} \bar{A}^ij)        
        self.Arr = 4.0/3.0 * (self.dWrdr - self.Wr / self.R)
        self.Att = 2.0/3.0 * (- R2 * self.dWrdr + self.R * self.Wr)
        self.App = 2.0/3.0 * (- R2 * self.dWrdr + self.R * self.Wr)
        
        #self.AijAij = (self.psi)**(-12.0) * (self.Arr * self.Arr / self.hrr / self.hrr 
        #                                  + self.Att * self.Att / self.htt / self.htt
        #                                  + self.App * self.App / self.hpp / self.hpp)
        
        self.AijAij = (self.psi)**(-12.0) * 24.0/9.0 *(self.Q - 3.0 * self.Wr/self.R)**2.0
    
    def get_Ham(self) :
        """Returns the value of the Ham constraint"""
    
        Ham_out = (self.K * self.K 
               - 1.5 * self.AijAij 
               - 3.0 * eight_pi_G * self.rho 
               - 12.0 * self.Lap_psi_over_psi5)
    
        return Ham_out

    def get_Mom(self) :
        """Returns the value of the Mom constraint"""

        csQ = CubicSpline(self.R, self.Q)
        dQdr = csQ(self.R, 1)
    
        csK = CubicSpline(self.R, self.K)
        dKdr = csK(self.R, 1)
    
        Mom_out = dQdr - 0.75 * eight_pi_G * self.psi**10.0 * self.SiU - 0.5 * self.psi**6.0 * dKdr
    
        return Mom_out