"""
GWSignal.py

Semi-analytical stochastic gravitational-wave (GW) spectrum from oscillons,
following Antusch, Cefalà & Orani (arXiv:1712.03231 )

Different parts of the code
A.  Per-run parameter extraction         -> class :class:OscillonRun

B.  Multi-oscillon GW spectrum integral  -> :func:Omega_GW

C.  Plotting / GR vs EsGB comparison     -> :func:compute_and_plot
                                            :func:compare_runs
                                            :func:plot_AGW_nGW_vs_w

D.  Analytic scaling helpers             -> :func:Omega_GW_amplitude_scaling
                                            :func:Omega_GW_asymmetry_scaling
                                            :func:Omega_GW_N_scaling
                                            :func:AGW_of_w, :func:nGW_of_w
###########
Units
###########

8 pi G = 1, hence M_Pl = 1
m_inflaton = scalar_mu = 1
Code Time =  units of 1/m and 
frequencies = units of m
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence

import numpy as np

# Matplotlib is imported lazily inside the plotting helpers so that this
# module can be imported in headless / non-graphical contexts.

# --------------------------------------------------------------------------- #
# Repository imports.  We follow the same pattern as `EFTdiagnosticz.ipynb`   #
# and add the project root to sys.path the first time this module is used.   #
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.grid import Grid                                      # noqa: E402
from core.spacing import CubicSpacing, NUM_GHOSTS               # noqa: E402
from core.statevector import StateVector                        # noqa: E402
from matter.scalarmatter_MG import ScalarMatter                 # noqa: E402
from backgrounds.sphericalbackground import FlatSphericalBackground  # noqa: E402
from bssn.bssnstatevariables import NUM_BSSN_VARS               # noqa: E402

# Some extra functions to fix renamings in different parts of the code
try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:                                  # pragma: no cover
    from scipy.integrate import cumtrapz as _cumtrapz  # type: ignore

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:                                                # pragma: no cover
    _trapz = np.trapz


# --------------------------------------------------------------------------- #
# Module-level configuration                                                  #
# --------------------------------------------------------------------------- #

# Range of ln(a/a0) over which we choose the oscillon to be "stable" after it has formed
# Hence all parameter extraction happens here, if we do earlier, oscillon might not be formed yet.
DEFAULT_STABLE_WINDOW = (2.0, 3.6)

# Minimum value of the inflaton field at the oscillon's potential minimum
PHI_MIN = 0.0

# ASSYMETRY PARAMETER VALUE
DEFAULT_DELTA = 0.3

# Default base data directory.  Falls back to ``$VSC_DATA/oscillon_runs``.
DEFAULT_DATA_DIR = os.path.join(
    os.environ.get("VSC_DATA", os.path.join(_PROJECT_ROOT, "RunningCode", "DATA")),
    "oscillon_runs",
)

# Default tags for the GR / EsGB runs used (can be changed to g2 runs as wel.)
DEFAULT_GR_TAG   = "lgb0.0_mu0.08_a0.2_b0.4_amp-0.02_R3_dr0.0625_quadratic_0"
DEFAULT_ESGB_TAG = "lgb1.0_mu0.08_a0.0_b0.0_amp-0.02_R3.0_dr0.041666666666666664_quadratic_0"


##############################################
# PART A: Data loader and parameter extracter
##############################################

@dataclass
class OscillonRun:
    """
    Loader + parameter extractor for a single simulation directory.
    """

    run_dir: str
    stable_window: Tuple[float, float] = DEFAULT_STABLE_WINDOW
    label: Optional[str] = None

    # populated lazily in __post_init__ / extract_all
    meta: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)
    t: np.ndarray = field(default=None, repr=False)
    r: np.ndarray = field(default=None, repr=False)
    grid: Grid = field(default=None, repr=False)
    matter: ScalarMatter = field(default=None, repr=False)
    background: FlatSphericalBackground = field(default=None, repr=False)

    # Cached internals
    _solution: Optional[np.ndarray] = field(default=None, repr=False)
    _u_r0_t: Optional[np.ndarray] = field(default=None, repr=False)
    _phi_r0_t: Optional[np.ndarray] = field(default=None, repr=False)

    # Extracted parameters (filled by extract_all)
    A: Optional[float] = None
    R: Optional[float] = None
    omega_source: Optional[float] = None
    harmonics: Optional[np.ndarray] = None
    w: Optional[float] = None
    H0: Optional[float] = None

    ################
    # Construction
    ################

    def __post_init__(self):
        if self.label is None:
            self.label = os.path.basename(os.path.normpath(self.run_dir))

        # Reconstructs the Grid and ScalarMatter structure from the 1D flattened array
        # This way we can run it as a diagnostic after the simulation is done.
        self.t   = np.load(os.path.join(self.run_dir, "t.npy"))
        self.r   = np.load(os.path.join(self.run_dir, "r.npy"))
        self.meta = dict(np.load(os.path.join(self.run_dir, "metadata.npz"),
                                 allow_pickle=True))
        diag_path = os.path.join(self.run_dir, "diagnostics.npz")
        if not os.path.exists(diag_path):
            raise FileNotFoundError(
                f"diagnostics.npz not found in {self.run_dir}.  "
                "Re-run get_oscillon_diagnostic to generate it."
            )
        self.diag = dict(np.load(diag_path, allow_pickle=True))

        self._build_grid_and_matter()

    def _build_grid_and_matter(self):
        """
        Reconstruct the same Grid / ScalarMatter objects used at run time.
        """
        scalar_mu       = 1.0
        selfinteraction = float(self.meta.get("selfinteraction", 0.08))
        r_max  = float(self.meta.get("r_max", 150.0))
        min_dr = float(self.meta.get("min_dr", 1.0 / 16.0))
        max_dr = float(self.meta.get("max_dr", 2.0))

        self.matter = ScalarMatter(scalar_mu, selfinteraction)
        sv          = StateVector(self.matter)
        spacing     = CubicSpacing(**CubicSpacing.get_parameters(r_max, min_dr, max_dr))
        self.grid       = Grid(spacing, sv)
        self.background = FlatSphericalBackground(self.grid.r)

    ##########
    # Helpers
    ##########

    @property
    def lambda_gb(self) -> float:
        """
        Gauss-Bonnet coupling lambda_GB (0 for GR runs).
        """
        return float(self.meta.get("lambda_gb", 0.0))

    @property
    def selfinteraction(self) -> float:
        """
        Inflaton scale mu (=scalar_mu * mu of the potential).
        """
        return float(self.meta.get("selfinteraction", 0.08))

    @property
    def is_GR(self) -> bool:
        return abs(self.lambda_gb) < 1e-12

    def _stable_mask(self) -> np.ndarray:
        """
        Boolean mask over time-steps inside stable_window of ln(a/a0).
        """
        ln_a = np.asarray(self.diag["ln_a"])
        lo, hi = self.stable_window
        return (ln_a >= lo) & (ln_a <= hi)

    def _solution_array(self) -> np.ndarray:
        """
        Lazy load of solution.npy.
        """
        if self._solution is None:
            self._solution = np.load(os.path.join(self.run_dir, "solution.npy"))
        return self._solution

    @property
    def u_r0_t(self) -> np.ndarray:
        """
        Scalar field at the innermost physical cell r[NUM_GHOSTS] vs t.

        This is the time series that feeds :meth:`extract_amplitude` and
        :meth:fft_source_spectrum. r0 != 0 (we sit at the first
        non-ghost cell, --min_dr/2) but the difference is negligible for
        the central-field amplitude / oscillation frequency.
        """
        if self._u_r0_t is None:
            sol = self._solution_array()
            num_vars = sol.shape[1] // self.grid.num_points
            sol_v = sol.reshape(sol.shape[0], num_vars, self.grid.num_points)
            self._u_r0_t = sol_v[:, NUM_BSSN_VARS, NUM_GHOSTS].copy()
            del sol_v
        return self._u_r0_t

    ######################
    # Extraction functions
    ######################

    # Gives us the amplitude in the inner most physical radius (that is not r=0)
    def extract_amplitude(self) -> float:
        """
        Eq. (paper Appendix A): A = max |phi(r0, t) - phi_min| in stable window.

        phi_min = 0 for the EsGB inflaton potential
        V = m^2 mu^2 / 2 * (1 - exp(phi/mu))^2 (vacuum at phi = 0).
        """
        mask = self._stable_mask()
        if not np.any(mask):
            return float(np.nanmax(np.abs(self.u_r0_t - PHI_MIN)))
        return float(np.nanmax(np.abs(self.u_r0_t[mask] - PHI_MIN)))

    def extract_R(self) -> float:
        """
        Reuse R from oscillon diagnostics.

        See oscillondiagnostic.py line 216:
        R = (3 V_proper / 4 pi)^{1/3}  with V_proper the volume of the
        ball where rho > 0.05 * rho_c.  We average over the stable window.

        This approach is more physical than the GW production paper who define radius 
        as 1/sqrt(e), but since the oscillons from simulations are not exactly gaussian
        The 5% diagnostic is more physical.
        """
        mask = self._stable_mask()
        R_arr = np.asarray(self.diag["R"])
        good = mask & np.isfinite(R_arr) & (R_arr > 0)
        if not np.any(good):
            good = np.isfinite(R_arr) & (R_arr > 0)
        return float(np.mean(R_arr[good]))

    # Function that will get us the oscillation frequency of the oscillons.
    def fft_source_spectrum(
        self,
        # number of overtones you want to include 
        n_harmonics: int = 5,
        prominence_ratio: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Power spectrum of Phi^2(t) with Phi = u_r0 - PHI_MIN.

        Note:
        The paper notes (p. 10) that, for symmetric potentials,
        Phi^2 oscillates at twice the oscillation frequency
        omega_osc.  For our asymmetric potential the dominant peak of
        Phi^2 is normally at omega_osc itself.  We just take the
        highest peak as omega_source regardless.
        """
        mask = self._stable_mask()
        if np.sum(mask) < 16:
            mask = np.ones_like(mask)

        # We select times in the stable window we choose earlier
        # and then substract the potential pinimum and square it 
        # we then remove the average, which leaves us only with the oscillating piece
        t   = self.t[mask]
        sig = (self.u_r0_t[mask] - PHI_MIN) ** 2
        sig = sig - np.mean(sig)                           # remove DC

        dt = float(np.mean(np.diff(t)))
        n  = sig.size

        # We produce a hann window [smoothly putting the signal to zero at edges]
        # to not have sharp jumps in frequency
        # at the end of the time series (fake frequencies -> leakage)
        win = np.hanning(n)
        sig_w = sig * win

        # Applying fast fourier transformation on the signal 
        # and extracting all frequencies out of it and the power (amp) per frequency 
        # to see which of those freqs is most dominant  
        freqs = np.fft.rfftfreq(n, d=dt)             

        # This is the raw FFT Power      
        amps  = np.abs(np.fft.rfft(sig_w)) ** 2
        # angular frequency (since these are the units from GW paper)
        omegas = 2.0 * np.pi * freqs                       
        # Skip the zero-frequency bin
        omegas, amps = omegas[1:], amps[1:]

        # Finding the local maxima in the powerspectrum (if its larger than neighbours)
        peaks = []
        if amps.size >= 3:
            local = (amps[1:-1] > amps[:-2]) & (amps[1:-1] > amps[2:])
            idx_local = np.where(local)[0] + 1
            if idx_local.size:
                amp_max = amps[idx_local].max()
                idx_strong = idx_local[amps[idx_local] > prominence_ratio * amp_max]
                order = np.argsort(amps[idx_strong])[::-1][:n_harmonics]
                for i in idx_strong[order]:
                    peaks.append((omegas[i], amps[i]))

        if not peaks:
            i = int(np.argmax(amps))
            peaks.append((omegas[i], amps[i]))

        return omegas, amps, np.asarray(peaks)

    # The function that is actually used
    # Throws away full freq axis, just keeps dominant freq and all other harmonics
    def extract_omega_source(
        self,
        n_harmonics: int = 5,
    ) -> Tuple[float, np.ndarray]:
        """
        Return the dominant peak omega_source and the harmonic table.
        """
        _, _, peaks = self.fft_source_spectrum(n_harmonics=n_harmonics)
        return float(peaks[0, 0]), peaks

    def extract_H0(self) -> float:
        """
        Hubble parameter H = -<K>/3 at the start of the stable window.
        No slope fit is performed here (like in the GW paper) the GW background cosmology assumes
        w = 0 (matter domination, see module docstring), so H0 is just
        the Hubble value at the chosen start time of the stable phase.
        """
        mask = self._stable_mask()
        K = np.asarray(self.diag["K_avg"])
        H_all = -K / 3.0
        good = mask & np.isfinite(H_all) & (H_all > 0)
        if not np.any(good):
            good = np.isfinite(H_all) & (H_all > 0)
            if not np.any(good):
                return 0.0
        return float(H_all[good][0])

    #################################################################################
    # EXTRA DIAGNOSTIC TO CHECK THE EOS parameter, not relevant for GW calculation
    #################################################################################
    def fit_eos_w(self) -> Tuple[float, float, float]:
        """
        Linear fit of ln H vs ln a -> equation-of-state w.

        This is a *diagnostic* helper used by :func: check_equation_of_state
        and is **no longer called** from :meth: extract_all. The GW pipeline
        assumes w = 0 regardless of what this fit returns.

        Returns
        -------
        w_fit : float
            Inferred equation of state, ``w = -1 - (2/3) d(ln H)/d(ln a)``.
        H0_fit : float
            ``H`` at the start of the stable window.
        r_squared : float
            Coefficient of determination of the linear fit (sanity check).
        """
        mask = self._stable_mask()
        ln_a = np.asarray(self.diag["ln_a"])
        K    = np.asarray(self.diag["K_avg"])
        H_all = -K / 3.0

        good = mask & (H_all > 0) & np.isfinite(H_all) & np.isfinite(ln_a)
        if np.sum(good) < 4:
            good = (H_all > 0) & np.isfinite(H_all) & np.isfinite(ln_a)
            if np.sum(good) < 4:
                return 0.0, float(H_all[np.argmax(good)] if np.any(good) else 0.0), 0.0

        x = ln_a[good]
        y = np.log(H_all[good])
        slope, intercept = np.polyfit(x, y, 1)
        # H ~ a^{-3(1+w)/2}  -->  d ln H / d ln a = -3 (1+w) / 2
        # so  w = -1 - (2/3) * d ln H / d ln a
        w_fit = -1.0 - 2.0 * slope / 3.0
        H0_fit = float(H_all[good][0])

        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-30
        r_squared = 1.0 - ss_res / ss_tot
        return float(w_fit), float(H0_fit), float(r_squared)

    #################################################################################

    # Extracts and caches all needed parameters from solution run to insert into GW equations
    def extract_all(self, verbose: bool = False) -> dict:
        """
        Compute the parameters that feed the GW pipeline and cache them.

        w is **hard-coded to 0** (matter domination); use
        :func: check_equation_of_state to verify this for your run.
        Returns a dict with keys A, R , omega_source harmonics
        w, H0, label,  lambda_gb, stable_window.
        """
        if self.A is None or self.R is None or self.omega_source is None:
            self.A = self.extract_amplitude()
            self.R = self.extract_R()
            self.omega_source, self.harmonics = self.extract_omega_source()
            self.H0 = self.extract_H0()
            self.w  = 0.0   # hard-coded; see check_equation_of_state for sanity check

        out = dict(
            A=self.A, R=self.R,
            omega_source=self.omega_source,
            harmonics=self.harmonics,
            w=self.w, H0=self.H0,
            label=self.label, lambda_gb=self.lambda_gb,
            stable_window=self.stable_window,
        )
        if verbose:
            print(f"[{self.label}]")
            print(f"  A             = {self.A:.4e}")
            print(f"  R             = {self.R:.4f}  (1/m)")
            print(f"  omega_source  = {self.omega_source:.4f}  (m)")
            print(f"  w             = {self.w:+.3f}   (assumed; not fitted -- see check_equation_of_state)")
            print(f"  H0            = {self.H0:.4e}  (m)")
        return out

    #PLOTTING FUNCTION
    def plot_source_power_spectrum(self, ax=None, n_harmonics: int = 5,
                                   color: str = "C0"):
        """
        Plot |FFT(Phi^2)|^2 vs angular frequency, mark the harmonics.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        omegas, amps, peaks = self.fft_source_spectrum(n_harmonics=n_harmonics)
        amps_norm = amps / max(amps.max(), 1e-30)
        ax.semilogy(omegas, amps_norm, color=color, lw=1.2, label=self.label)
        for i, (omega_p, amp_p) in enumerate(peaks):
            ax.axvline(omega_p, color=color, ls="--", alpha=0.4, lw=0.8)
            ax.annotate(rf"$\omega_{{{i+1}}}={omega_p:.2f}$",
                        xy=(omega_p, amp_p / max(amps.max(), 1e-30)),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=color, alpha=0.9, va="center")

        ax.set_xlabel(r"$\omega \;[m]$", fontsize=12)
        ax.set_ylabel(r"$|\widetilde{\Phi^2}(\omega)|^2$ (normalised)", fontsize=12)
        ax.set_title(rf"Power spectrum of $\Phi^2(t) = (\phi(r_0,t)-\phi_{{\min}})^2$  "
                     rf"(stable window {self.stable_window})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        return ax


##########################################################
# EOS parameter checker, not needed for GW calculatino
##########################################################

def check_equation_of_state(
    sim_data,
    *,
    n_cycles: float = 5.0,
    tol: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Sanity-check ``w`` from ``H(a)`` for a single simulation run.

    Parameters
    ----------
    sim_data : :class:`OscillonRun` or path-like
        Either an existing :class:`OscillonRun` or a run directory / tag that
        :class:`OscillonRun` knows how to resolve.
    n_cycles : float, default 5
        Approximate number of oscillation cycles ``T_osc = 2 pi / omega_source``
        used as the smoothing window when computing the *local* slope.  The
        global ``w_eff`` is fitted across the full stable window irrespective
        of this knob.
    tol : float, default 0.05
        Threshold above which we print a warning (``|w_eff| > tol``).
    verbose : bool
        Print a short report.

    Method
    ------
    Loads ``a(t)`` from ``ln_a`` and ``H(t) = -<K>/3`` from ``K_avg``, both
    stored in ``diagnostics.npz``, restricted to the run's stable window.
    A linear fit of ``ln H`` vs ``ln a`` gives the slope ``s``, and the
    inferred equation of state is

    .. math::
       w_{\\rm eff} = -1 - \\frac{2}{3}\\, \\frac{d \\ln H}{d \\ln a},

    which is the inversion of ``H \\propto a^{-3(1+w)/2}`` for an FLRW fluid.
    For matter domination (``w = 0``) the slope is ``-3/2``.

    .. note::
       The user prompt wrote ``w_eff = (2/3) d ln H/d ln a - 1``; that is a
       sign-flip typo (it would give ``w_eff = -2`` in matter domination).
       The implementation here uses the physically correct inversion.

    Returns
    -------
    dict with keys:
      * ``w_eff``     : global linear-fit estimate (full stable window)
      * ``w_local``   : array of local estimates (sliding window of ``n_cycles``)
      * ``ln_a``, ``H``: arrays restricted to the stable window
      * ``r2``        : R^2 of the linear fit
      * ``warning``   : True iff ``|w_eff| > tol``
    """
    if isinstance(sim_data, OscillonRun):
        run = sim_data
    elif isinstance(sim_data, str):
        run = OscillonRun(run_dir=sim_data)
    else:
        raise TypeError(f"sim_data must be OscillonRun or path-like, "
                        f"got {type(sim_data).__name__}")

    w_fit, H0_start, r2 = run.fit_eos_w()

    mask = run._stable_mask()
    ln_a = np.asarray(run.diag["ln_a"])
    K    = np.asarray(run.diag["K_avg"])
    H_all = -K / 3.0
    good = mask & np.isfinite(ln_a) & np.isfinite(H_all) & (H_all > 0)
    ln_a_w = ln_a[good]
    H_w    = H_all[good]
    ln_H_w = np.log(H_w)

    # Local slope on a sliding window of ~ n_cycles oscillation periods.
    # The stable window is sampled at the diagnostic cadence; we map
    # n_cycles * T_osc to a number of samples via t = a / (a * H) ~ ln(a)/H,
    # but the simplest robust thing is just to use a fraction of the window.
    w_local = np.array([])
    if ln_a_w.size >= 8:
        try:
            omega_s, _ = run.extract_omega_source()
        except Exception:
            omega_s = 0.0
        # Span of the stable window in cosmic time, approximated as
        # Delta_t ~ Delta(ln a) / H_avg.
        if omega_s > 0:
            T_osc = 2.0 * np.pi / omega_s
            H_avg = float(np.mean(H_w))
            d_lna = ln_a_w[-1] - ln_a_w[0]
            dt    = d_lna / max(H_avg, 1e-30)
            n_samples_per_window = max(4, int(round(n_cycles * T_osc / dt * ln_a_w.size)))
            n_samples_per_window = min(n_samples_per_window, ln_a_w.size // 2)
        else:
            n_samples_per_window = max(4, ln_a_w.size // 8)
        half = n_samples_per_window // 2
        slopes = []
        for i in range(half, ln_a_w.size - half):
            xs = ln_a_w[i - half:i + half + 1]
            ys = ln_H_w[i - half:i + half + 1]
            if xs.size >= 3 and np.ptp(xs) > 0:
                s, _ = np.polyfit(xs, ys, 1)
                slopes.append(-1.0 - 2.0 * s / 3.0)
        w_local = np.asarray(slopes)

    warn = abs(w_fit) > tol

    if verbose:
        print(f"[check_equation_of_state] {run.label}")
        print(f"  stable window      = ln a in {run.stable_window}")
        print(f"  H0 (start of win.) = {H0_start:.4e}  m")
        print(f"  d ln H / d ln a    = {(-3.0/2.0)*(1.0 + w_fit):+.4f}")
        print(f"  w_eff (global fit) = {w_fit:+.4f}   (R^2 = {r2:.3f})")
        if w_local.size:
            print(f"  w_eff (local)      = {np.mean(w_local):+.4f}  "
                  f"+/- {np.std(w_local):.4f}   "
                  f"({w_local.size} sliding windows of ~{n_cycles:.1f} cycles)")
        if warn:
            print(f"  WARNING: |w_eff| = {abs(w_fit):.4f} > {tol:.2f} -- the "
                  f"matter-domination assumption (w=0) used by Omega_GW may be "
                  f"inaccurate for this run.")
        else:
            print(f"  OK: |w_eff| <= {tol:.2f}; w=0 (matter dom.) is a fine "
                  f"assumption for the GW pipeline.")

    return dict(
        w_eff=float(w_fit),
        w_local=w_local,
        ln_a=ln_a_w,
        H=H_w,
        r2=float(r2),
        warning=bool(warn),
        H0=float(H0_start),
        label=run.label,
    )
##########################################################
##########################################################


############################################################
#PART B: Semi-analytical multi-oscillon GW spectrum                    #
############################################################

# Here we compute how much the universe (scale factor) has expanded at each time step
# the scale factor evolves depending on our EOS parameter w
# calculated based on time array, w and H0.
def background_a_of_t(t: np.ndarray, w: float, H0: float) -> np.ndarray:
    """
    Background scale factor a(t) for a fluid with EoS w 
    Equation 4.13 form the GW paper
    """
    # special case (to avoid blowing up when w=-1)
    if abs(1.0 + w) < 1e-6:
        return np.exp(H0 * (t - t[0]))
    # general case
    p = 2.0 / (3.0 * (1.0 + w))
    base = (2.0 / 3.0) + H0 * (1.0 + w) * t
    base = np.where(base > 0, base, 1e-30)
    pref = (9.0 / 4.0) ** (1.0 / (3.0 * (1.0 + w)))
    return pref * base ** p

# Computes the cosmic time (simulation time t) to conformal time (time in GW eqs)
# we do a numerical integration tau= \int dt/a(t)
# When we calculate the GW spectrum, we use conformal time (to scale out expansion)
def conformal_time(t: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Cumulative conformal time tau(t) = int_0^t dt'/a(t').
    """
    inv_a = 1.0 / np.where(a > 0, a, 1e-30) #safeguard against zeros
    tau = _cumtrapz(inv_a, t, initial=0.0)
    return tau

# Randomly puts N oscillons in a box of Size L^3.
# For me not really needed since spherical symmetry.
# But could be intresting to study interference effects 
# of oscillons at different locations.
# With randomly distributed oscillons the GW signal scales as
# \Omega^N_GW =  N \Omega^1_GW
def _draw_random_positions(N: int, R: float, V: float,
                           d_min_factor: float = 4.0,
                           rng: np.random.Generator = None,
                           max_tries: int = 10_000) -> np.ndarray:
    """
    Random oscillon positions in box [0, L]^3 with L = V^{1/3},
    rejecting any configuration whose minimum pairwise distance is below
    d_min_factor * R (paper eq. 4.5).
    """
    if rng is None:
        rng = np.random.default_rng()
    L = V ** (1.0 / 3.0)
    d_min = d_min_factor * R
    positions = np.empty((N, 3))
    for q in range(N):
        for _ in range(max_tries):
            cand = rng.uniform(0.0, L, size=3)
            if q == 0:
                positions[q] = cand
                break
            dists = np.linalg.norm(positions[:q] - cand, axis=1)
            if np.min(dists) >= d_min:
                positions[q] = cand
                break
        else:
            # Failed to place this oscillon: just put it down anyway
            positions[q] = cand
    return positions


def _f_ij_polarisation(k_hat: np.ndarray) -> np.ndarray:
    """
    f_ij(k) Captures how the GW emission is distributed across 
    directions and polarizations for a given wavevector K.
    [Which results from applying the TT projection to the anisotropic 
    part of the stress tensor.]

    f_ij(k) from paper eq. 3.11.
    Returns a real symmetric 3x3 matrix:
    f_ij is well-defined and symmetric.
    """
    kx, ky, kz = k_hat
    # outer product
    kk = np.outer(k_hat, k_hat)
    delta = np.eye(3)

    # picking out the unit y direction
    # since this is our chosen direction for the assymetry 
    ey = np.array([0.0, 1.0, 0.0])
    ey_kron = np.outer(ey, ey)

    # symmetrized product scaled by ky
    sym_ky_k = ky * (np.outer(ey, k_hat) + np.outer(k_hat, ey))
    # factor 1/2 takes into account the extra factor of 2 from the symmetrization.
    f = ((kk - delta) 
            - (ky ** 2) * (kk - delta) 
            - sym_ky_k + (ky ** 2) * delta 
            - ey_kron + 2.0 * (ky ** 2) * ey_kron)
    return 0.5 * (f + f.T)

# The machine that takes any 3×3 tensor and extracts only the gravitational wave part.
# or in other words takes teh transverse traceless projection
def _Lambda_TT(k_hat: np.ndarray) -> np.ndarray:
    """
    Transverse-traceless projector Lambda_{ij,lm}(k_hat)
    as a 3x3x3x3 tensor.
    """
    # Transverse projection, removing the component along k
    P = np.eye(3) - np.outer(k_hat, k_hat)    
    # Equation 2.11 from GW paper
    # \Lambda_ij, lm = \frac{1}{2}(Pil Pjm + Pim Pjl - Pij Plm)
    L = 0.5 * (np.einsum("il,jm->ijlm", P, P)
                + np.einsum("im,jl->ijlm", P, P)
                - np.einsum("ij,lm->ijlm", P, P))
    return L

# basicly calculates full anisotropic stress tensor T^TT 
# without the fijk (because it is calculated above).
def T_TT_single_envelope(
    k_vec: np.ndarray,
    a_t:   np.ndarray,
    R:     float,
    Delta: float,
) -> np.ndarray:
    """
    Spatial part of the SINGLE oscillon GW source, GW paper eq. 3.10.
    Returns:
    env : ndarray, shape (len(k_vec), len(a_t))
    """
    # Ensures the correct shape
    k_vec = np.atleast_2d(k_vec)
    kx, ky, kz = k_vec[..., 0], k_vec[..., 1], k_vec[..., 2]
    
    # Anisotropic combination (INTRODUCTION OF ASSYMETRY)
    k_anis_sq = kx[:, None] ** 2 + kz[:, None] ** 2 + (1.0 + Delta) ** 2 * ky[:, None] ** 2
    # reshaping for correct broadcasting
    a_t = np.atleast_1d(a_t)[None, :]
    # numerical prefactors
    pref = (np.pi ** 1.5) * Delta * (Delta + 2.0) * R / (4.0 * a_t * (Delta + 1.0))
    return pref * np.exp(-R ** 2 * k_anis_sq / (4.0 * a_t ** 2))

# How much GW energy is carried by a GW at each wavenumber k
# This function tells us how to calculate one specific point in the spectrum
# The next function will be the loop over the full spectrum 
def Omega_GW(
    A: float,
    R: float,
    omega_source: float,
    H0: float,
    *,
    w: float = 0.0,
    Delta: float = DEFAULT_DELTA,
    N: int = 1,
    V: Optional[float] = None,
    n_angles: int = 30,
    n_tau: int = 1500,
    k_grid: Optional[np.ndarray] = None,
    tau_window: Optional[Tuple[float, float]] = None,
    seed: int = 42,
    harmonics: str = "cosine",
    fourier_modes: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> dict:
    
    """
    Compute the dimensionless GW power spectrum, GW paper eq. 3.18.

    * a(t) and tau(t) come from eqs. 4.13 with the extracted w and
      H0.  The integration window defaults to [0, 100/H0]
      (long enough to enclose any oscillon emission, 
      short enough to keep the sampling tractable).


    * N random oscillon positions are drawn in a comoving box of volume
      V (default (100 R)^3) with rejection so that every pairwise
      distance>= 4 R (eq. 4.5).  Random phases phi_q are drawn
      uniformly in [0, 2 pi).


    * The spherical solid-angle integral of eq. 3.18 is discretised as a sum
      over an n_angles x n_angles grid in (theta, phi) (eqs. 3.19-3.20).

    Parameters
    ----------
    A, R, omega_source, H0 Oscillon and background parameters extracted by :class:`OscillonRun`.
    
    w : float, keyword-only, default 0.0
        Equation of state of the background. 

    Delta : float
        Asymmetry parameter (free parameter of the model, paper Sec. 4.2).

    N : int
        Number of oscillons in the box.

    V : float, optional
        Comoving volume of the box (defaults to (100 R)^3).

    n_angles : int
        Resolution of the (theta, phi) grid (paper uses 30).

    n_tau : int
        Number of conformal-time samples used for the time integral.

    k_grid : ndarray, optional
        Comoving |k| grid.  Defaults to a logarithmic grid spanning
        [0.05, 5] * omega_source.

    tau_window : tuple, optional
        (t_start, t_end) of the cosmic-time window for the source.  Defaults
        to (0, 100/H0) (no impact on the answer beyond statistical noise).

    seed : int
        RNG seed for reproducibility.

    harmonics : {'cosine', 'fourier'}
        ``'cosine'`` (default) uses ``Phi(t) = A cos(omega_source t + phi_q)``.
        ``'fourier'`` uses the harmonic decomposition supplied via
        ``fourier_modes`` -- a sequence of ``(omega_n, amp_n, phase_n)`` tuples
        such that ``Phi(t) = sum_n amp_n cos(omega_n t + phase_n)``.

    Returns:
      * k            -- comoving |k| grid (1/length, code units)
      * k_phys       -- physical wavenumber today: k / a_f
      * Omega_GW     -- dimensionless GW spectrum at a_f
      * peak_k       -- k_phys of the spectrum maximum
      * A_GW, n_GW   -- power-law fit Omega_GW = A_GW (k/omega_s)^{n_GW}
        in the rising part of the spectrum.
      * a_f, H_f, rho_c -- final scale factor, Hubble, critical density
      * positions, phases -- the random ensemble used.
    """
    rng = np.random.default_rng(seed)

    # A cosmological grid is chosen, 
    # 1/H0 is one hubble time, time scale of the universe at the moment oscillons form
    # it is the timescale in which the universe dubbles in size
    # 100/H0: we are integrating the GW source over 100 hubble times 
    # meaning that the source shuts itself off, since after a few hubble times the 
    # comoving size of the oscillons shrujnk so much that the contribution to GW is negligable

    # 100 hublle times is maybe to conservative, 10 is more realistic. will be tested later
    # UPDATE: 3 seems to be even more plausible
    if tau_window is None:
        t_end = 100.0 / max(H0, 1e-30)
        tau_window = (0.0, t_end)
    
    # making cosmo grid
    t_grid = np.linspace(tau_window[0], tau_window[1], n_tau)
    a_grid = background_a_of_t(t_grid, w, H0)
    tau_grid = conformal_time(t_grid, a_grid)

    # final scale factor, hubble parameter and critical density (at end of integration)
    # needed for normalization of \Omeaga_GW
    a_f = a_grid[-1]
    H_f = H0 * (a_f) ** (-1.5 * (1.0 + w)) if abs(1.0 + w) > 1e-6 else H0
    M_Pl = 1.0
    rho_c = 3.0 * H_f ** 2 * M_Pl ** 2

    # Volume of box and position (will be irrelevant when N=1)
    if V is None:
        V = (100.0 * R) ** 3
    positions = _draw_random_positions(N, R, V, rng=rng)
    phases    = rng.uniform(0.0, 2.0 * np.pi, size=N)

    # HIGHER RESOLUTION IN K AXIS
    # k grid, the k values at which the Omega_GW will be evaluated
    # Here you can insert more than 40 points for higher resolution
    if k_grid is None:
        k_grid = np.logspace(np.log10(0.05 * omega_source),
                             np.log10(5.0 * omega_source), 180)
                             # HERE ABOVE YOU CAN ADJUST THE K_GRID MAXIMUM <- PrevVal: 40
    k_grid = np.asarray(k_grid)
    nk = k_grid.size

    # Setting up the Angular grid (solid angle integral discretization) 3.18
    theta = np.linspace(0.5 * np.pi / n_angles, np.pi - 0.5 * np.pi / n_angles, n_angles)
    phi   = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    dth = np.pi / n_angles
    dph = 2.0 * np.pi / n_angles
    sin_th = np.sin(theta)

    # Harmonics (overtones), the higher freq components of a periodic signal
    # since the potential is not perfect quadratic (not pure cos(x)), there is more than one pure mode.
    # So you have a fundamental term (corresponds to quadratic) and then higher corrections (which are the harmnoics)
    # phi(t) = c1 cos(omega t) + c2 cos(2omega t) + c3 cos(3 omegat t) + ...
    
    # Cosine mode keeps only the fundamental mode 
    # only gives the primary peak in the spectrum
    if harmonics == "cosine":
        # Phi(t) = A cos(omega t + phi_q)  =>  Phi^2 = A^2 cos^2 = A^2/2 (1 + cos(2 omega t + 2 phi_q))
        modes = [(omega_source, A, 0.0)]
    
    # Fourier mode includes the harmonics from the FFT analysis
    # Also gives the secondary peaks in the spectrum
    # essentially adds a sum of cosines with different phases to the primary mode
    elif harmonics == "fourier":
        if fourier_modes is None or len(fourier_modes) == 0:
            raise ValueError("harmonics='fourier' requires `fourier_modes`.")
        modes = list(fourier_modes)
    else:
        raise ValueError(f"Unknown harmonics={harmonics!r}")
    
    ##################################################################
    # Start precompute tabulated Lambda(k_hat) for each angle outside big loop
    Lambda_tab = np.empty((n_angles, n_angles, 3, 3, 3, 3))
    f_tab      = np.empty((n_angles, n_angles, 3, 3))
    khat_tab   = np.empty((n_angles, n_angles, 3))

    # loop over all angles
    for it in range(n_angles):
        st, ct = np.sin(theta[it]), np.cos(theta[it])
        for ip in range(n_angles):
            sp, cp = np.sin(phi[ip]), np.cos(phi[ip])

            #Building unit k vector in spherical coordinates
            kh = np.array([st * cp, st * sp, ct])

            #storing multiple objects per direction
            khat_tab[it, ip] = kh
            Lambda_tab[it, ip] = _Lambda_TT(kh)
            f_tab[it, ip] = _f_ij_polarisation(kh)
    # End precompute
    ##################################################################
   
    ##################################################################
    # START OF BIG CALCULATION LOOP (calculating 3.18 numerically)

    Omega = np.zeros(nk)
    
    # for each k value
    for ik, k in enumerate(k_grid):
        Omega_k = 0.0
        # we integrate over all directions theta and phi
        for it in range(n_angles):
            for ip in range(n_angles):
                k_hat = khat_tab[it, ip]
                k_vec = (k * k_hat)[None, :]                         # shape (1, 3)
                Lambda = Lambda_tab[it, ip]
                fij    = f_tab[it, ip]

                # Single-oscillon spatial Fourier envelope (eq. 3.10)
                # this is time dependant trough a(t)
                env = T_TT_single_envelope(k_vec, a_grid, R, Delta)[0]   

                # The phase each oscillon would pick up due to different positions 
                # when they are randomly scattered on a grid 
                # In case of N=1, this is just a complex number
                kdotx = positions @ (k * k_hat)
                phase_factor = np.exp(-1j * kdotx)                       

                # Collapse harmonic decomposition of Phi^2(t)

                # IN THE CASE WE ARE USING COSINE MODE (only dominant contribution)
                # Phi_q^2 = A^2 cos^2(omega t + phi_q)
                if harmonics == "cosine":
                    omega_n = modes[0][0]
                    A_n     = modes[0][1]
                    # cos(omega t + phi_q)^2 = 0.5 (1 + cos(2 omega t + 2 phi_q))
                    phi_sq_q = 0.5 * A_n ** 2 * (1.0 + np.cos(2.0 * omega_n * t_grid[None, :]
                                                    + 2.0 * phases[:, None]))   
                
                # IN THE CASE WE ARE USING FOURIER MODE
                # General Fourier sum -> we just sum products
                else:
                    # Phi_q(t) = sum_n A_n cos(omega_n t + phi_q + phase_n)
                    Phi_q_t = np.zeros((N, n_tau))
                    for (omega_n, A_n, phase_n) in modes:
                        Phi_q_t += A_n * np.cos(omega_n * t_grid[None, :]+ phase_n+ phases[:, None])
                    phi_sq_q = Phi_q_t ** 2



                # Combining Source and structure factor
                # calculates the sum of all amplitudes squared, weighted by phase factor (weighting by position and phase)
                multi_sum_t = (phi_sq_q.T * phase_factor[None, :]).sum(axis=1)  
                # multiplying it by spatial enveloppe
                S_t = env * multi_sum_t                                          
                
                T_lm = S_t       
                # Here we are contracting \Lambda_{ij, lm} with f_ml to get the 3x3 directional polarization matrix
                # This is a time independent object and only depends on the direction k
                Lambda_f = np.einsum("ijlm,lm->ij", Lambda, fij)              
                
                # The full TT source tensor is 
                # T^TT_ij(tau) = Lambda_f_ij * S_t(tau)
                # Which is possible SINCE WE ASSUME THAT EACH OF THE N OSCILLONS HAS EQUAL SIZE 
                
                # We evaluate the Cij and Sij integrals
                cos_int = _trapz(np.cos(k * tau_grid)[None, :] * a_grid[None, :] * (Lambda_f[..., None] * T_lm.real[None, None, :]),tau_grid, axis=-1)                              
                sin_int = _trapz(np.sin(k * tau_grid)[None, :] * a_grid[None, :] * (Lambda_f[..., None] * T_lm.real[None, None, :]),tau_grid, axis=-1)
                cos_int_im = _trapz(np.cos(k * tau_grid)[None, :] * a_grid[None, :] * (Lambda_f[..., None] * T_lm.imag[None, None, :]),tau_grid, axis=-1)
                sin_int_im = _trapz(np.sin(k * tau_grid)[None, :] * a_grid[None, :] *(Lambda_f[..., None] * T_lm.imag[None, None, :]),tau_grid, axis=-1)
                
                # Take the modulus squared
                # |Cij|^2 = Re(Cij)^2 + Im(Cij)^2
                # The .sum() adds up all 9 components of (i,j) and gies the single number for 
                # sum of |Cij|^2+|Sij|^2 for this specific combination of (k, theta, phi)
                mod_sq = (cos_int ** 2 + cos_int_im ** 2 + sin_int ** 2 + sin_int_im ** 2).sum()

                # while the loop is running, we keep adding the contributions from each angular bin
                Omega_k += sin_th[it] * dth * dph * mod_sq
        # Adding prefactors as well
        Omega[ik] = ((k ** 3) / (2.0 * a_f ** 4 * rho_c) * (1.0 / V) * Omega_k / (2.0 * np.pi) ** 3)
    
    # END OF BIG CALCULATION LOOP (calculating 3.18 numerically)
    ##################################################################
    
    ##################################
    # Power law fit, Figure 3 of paper
    ##################################
    A_GW, n_GW, peak_k_phys = _fit_power_law(k_grid / a_f, Omega, omega_source)

    return dict(
        k=k_grid,
        k_phys=k_grid / a_f,
        Omega_GW=Omega,
        peak_k=peak_k_phys,
        A_GW=A_GW,
        n_GW=n_GW,
        a_f=a_f,
        H_f=H_f,
        rho_c=rho_c,
        positions=positions,
        phases=phases,
        params=dict(A=A, R=R, Delta=Delta, w=w, H0=H0, N=N, V=V,
                    omega_source=omega_source, harmonics=harmonics,
                    n_angles=n_angles, n_tau=n_tau, seed=seed),
    )


def _fit_power_law(k_phys: np.ndarray, Omega: np.ndarray, omega_source: float,
                   width: float = 1.0):
    """
    Fit Omega_GW = A_GW (k/omega_s)^{n_GW} on the rising side of the peak.
    """
    Omega = np.asarray(Omega)
    k_phys = np.asarray(k_phys)
    if not np.any(Omega > 0):
        return float("nan"), float("nan"), float("nan")
    i_peak = int(np.argmax(Omega))
    peak_k = float(k_phys[i_peak])
    # Choose a window below the peak for the fit
    fit_mask = (k_phys < peak_k) & (Omega > 0)
    if np.sum(fit_mask) < 3:
        return float("nan"), float("nan"), peak_k
    x = np.log(k_phys[fit_mask] / max(omega_source, 1e-30))
    y = np.log(Omega[fit_mask])
    n_GW, lnA = np.polyfit(x, y, 1)
    return float(np.exp(lnA)), float(n_GW), peak_k

############################################################
#PART C: Plotting GR vs EsGB + helper stuff
############################################################

# Full pipeline function
def compute_and_plot(
    run: OscillonRun,
    *,
    Delta: float = DEFAULT_DELTA,
    N: int = 1,
    label: Optional[str] = None,
    ax=None,
    color: str = "C0",
    spec_kwargs: Optional[dict] = None,
    **kw,
):
    """
    1) Extract parameters from run 
    2) compute Omega_GW(k_phys)
    3) plot it.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if spec_kwargs is None:
        spec_kwargs = {}
    params = run.extract_all()
    res = Omega_GW(
        A=params["A"], R=params["R"],
        omega_source=params["omega_source"],
        H0=params["H0"],   # w defaults to 0 (matter dom.), see check_equation_of_state
        Delta=Delta, N=N, **spec_kwargs, **kw,
    )
    label = label or run.label

    # We normalize the x axis k_phys / omega_src [resonance region is where this is \approx 1]
    # log log is double log scale, hence we are working with a power law
    ax.loglog(res["k_phys"] / params["omega_source"], res["Omega_GW"],
              color=color, lw=1.5, label=label)
    ax.set_xlabel(r"$k_{\rm phys}/\omega_{\rm source}$", fontsize=12)
    ax.set_ylabel(r"$\Omega_{\rm GW}(k)$", fontsize=12)
    ax.set_title(r"Stochastic GW spectrum from oscillons (paper eq. 3.18)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    return ax, res

# Compares GR vs Modified GR run
def compare_runs(
    run_GR: OscillonRun,
    run_EsGB: OscillonRun,
    *,
    labels: Tuple[str, str] = ("GR", "EsGB"),
    Delta: float = DEFAULT_DELTA,
    N: int = 1,
    seed: int = 42,
    spec_kwargs: Optional[dict] = None,
    figure_kwargs: Optional[dict] = None,
):
    """
    Overlay GR and EsGB Omega_GW(k) and their ratio
    Ratio is usefull for 

    Ratio > 1 MG produces more GW
    Ratio < 1 GR produces more GW
    """
    import matplotlib.pyplot as plt
    if spec_kwargs is None:
        spec_kwargs = {}
    if figure_kwargs is None:
        figure_kwargs = {}

    # Extract respective parameters from selected runs
    params_GR   = run_GR.extract_all()
    params_EsGB = run_EsGB.extract_all()

    # w defaults to 0 
    # EOS can be checked with "check_equation_of_state" to see if its close to w=0.
    
    # Calculate GR omega
    res_GR = Omega_GW(
        A=params_GR["A"], R=params_GR["R"],omega_source=params_GR["omega_source"],H0=params_GR["H0"],
        Delta=Delta, N=N, seed=seed, **spec_kwargs,)
    # Calculate MGR omega
    res_EsGB = Omega_GW(A=params_EsGB["A"], R=params_EsGB["R"],omega_source=params_EsGB["omega_source"],H0=params_EsGB["H0"],
        Delta=Delta, N=N, seed=seed, **spec_kwargs,)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8),
                             gridspec_kw=dict(height_ratios=[3, 1.5], hspace=0.05),
                             sharex=True, **figure_kwargs)
    ax_top, ax_bot = axes

    x_GR   = res_GR["k_phys"]   / params_GR["omega_source"]
    x_EsGB = res_EsGB["k_phys"] / params_EsGB["omega_source"]
    ax_top.loglog(x_GR,   res_GR["Omega_GW"],   "C0-",  lw=1.6, label=labels[0])
    ax_top.loglog(x_EsGB, res_EsGB["Omega_GW"], "C3--", lw=1.6, label=labels[1])
    ax_top.set_ylabel(r"$\Omega_{\rm GW}(k)$", fontsize=12)
    ax_top.set_title(r"GW spectrum: GR vs EsGB")
    ax_top.grid(True, alpha=0.3, which="both")
    ax_top.legend(fontsize=10)

    # Interpolate EsGB onto the GR x-axis to take the ratio
    Omega_EsGB_on_GR = np.interp(x_GR, x_EsGB, res_EsGB["Omega_GW"], left=np.nan, right=np.nan)
    ratio = Omega_EsGB_on_GR / np.where(res_GR["Omega_GW"] > 0,res_GR["Omega_GW"], np.nan)
    ax_bot.semilogx(x_GR, ratio, "k-", lw=1.4)
    ax_bot.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax_bot.set_xlabel(r"$k_{\rm phys}/\omega_{\rm source}$", fontsize=12)
    ax_bot.set_ylabel(r"$\Omega_{\rm GW}^{\rm EsGB}/\Omega_{\rm GW}^{\rm GR}$",fontsize=11)
    ax_bot.grid(True, alpha=0.3, which="both")
    
    # gridspec_kw already controls spacing; skip tight_layout to avoid warnings
    return fig, (ax_top, ax_bot), (res_GR, res_EsGB)

# Reproduces figure 6 and 7 from paper, GW in function of w (in case it would be needed)
def plot_AGW_nGW_vs_w(
    runs: Sequence[OscillonRun],
    ax=None,
    *,
    w_values: Optional[Sequence[float]] = None,
    show_paper_fit: bool = True,
    Delta: float = DEFAULT_DELTA,
    N: int = 1,
    spec_kwargs: Optional[dict] = None,
):
    """
    Plot fitted A_GW and n_GW vs w.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = ax[0].figure if isinstance(ax, (list, tuple)) else ax.figure
        axes = ax if isinstance(ax, (list, tuple)) else (ax, ax.twinx())
    ax_A, ax_n = axes
    if spec_kwargs is None:
        spec_kwargs = {}
    if w_values is None:
        w_values = [0.0]
    w_values = np.asarray(w_values, dtype=float)

    w_vals, A_vals, n_vals, lbls = [], [], [], []
    for run in runs:
        p = run.extract_all()
        for w_i in w_values:
            res = Omega_GW(A=p["A"], R=p["R"], omega_source=p["omega_source"],
                           H0=p["H0"], w=float(w_i),
                           Delta=Delta, N=N, **spec_kwargs)
            if np.isfinite(res["A_GW"]) and np.isfinite(res["n_GW"]):
                w_vals.append(float(w_i))
                A_vals.append(res["A_GW"])
                n_vals.append(res["n_GW"])
                lbls.append(f"{run.label} (w={w_i:+.2f})")

    w_vals = np.asarray(w_vals)
    A_vals = np.asarray(A_vals)
    n_vals = np.asarray(n_vals)

    ax_A.semilogy(w_vals, A_vals, "o", color="C0", label="numerical")
    ax_n.plot(w_vals, n_vals, "s", color="C3", label="numerical")

    if show_paper_fit:
        ws = np.linspace(min(w_vals.min(), -0.1) - 0.05,
                         max(w_vals.max(),  0.5) + 0.05, 100)
        ax_A.semilogy(ws, AGW_of_w(ws), "k--", lw=1.0, label="paper eq. 4.18")
        ax_n.plot(ws, nGW_of_w(ws), "k--", lw=1.0, label="paper eq. 4.20")

    ax_A.set_xlabel(r"$w$"); ax_A.set_ylabel(r"$A_{\rm GW}$")
    ax_A.set_title(r"GW amplitude $A_{\rm GW}(w)$")
    ax_A.grid(True, alpha=0.3); ax_A.legend(fontsize=9)
    ax_n.set_xlabel(r"$w$"); ax_n.set_ylabel(r"$n_{\rm GW}$")
    ax_n.set_title(r"Spectral tilt $n_{\rm GW}(w)$")
    ax_n.grid(True, alpha=0.3); ax_n.legend(fontsize=9)
    fig.tight_layout()
    return fig, axes, dict(w=w_vals, A_GW=A_vals, n_GW=n_vals, labels=lbls)

#################################
#PART D: Analytic scaling laws
#################################

def Omega_GW_amplitude_scaling(Omega_ref: np.ndarray, A: float, A_ref: float):
    r"""Section 4 scaling: :math:`\\Omega_{\\rm GW} \\propto A^4`."""
    return Omega_ref * (A / A_ref) ** 4


def Omega_GW_asymmetry_scaling(Omega_ref: np.ndarray, Delta: float, Delta_ref: float):
    r"""Section 4 scaling: :math:`\\Omega_{\\rm GW} \\propto \\Delta^2`."""
    return Omega_ref * (Delta / Delta_ref) ** 2


def Omega_GW_N_scaling(Omega_ref: np.ndarray, N: int, N_ref: int):
    r"""Section 4 scaling for incoherent random phases: :math:`\\Omega_{\\rm GW} \\propto N`."""
    return Omega_ref * (N / N_ref)


def AGW_of_w(w, alpha: float = 5.68e-7, kappa: float = 9.46):
    """Analytic GW amplitude (paper eq. 4.18): ``A_GW(w) = alpha exp(kappa w)``."""
    return alpha * np.exp(kappa * np.asarray(w))


def nGW_of_w(w, beta: float = 2.49, gamma: float = 1.55):
    """Analytic spectral tilt (paper eq. 4.20): ``n_GW(w) = beta + gamma w``."""
    return beta + gamma * np.asarray(w)


# --------------------------------------------------------------------------- #
# Convenience: discover runs by tag prefix                                    #
# --------------------------------------------------------------------------- #


def find_run_dir(tag_or_prefix: str, data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Return ``os.path.join(data_dir, tag_or_prefix)`` if it exists, else search.

    Falls back to the first sub-directory of ``data_dir`` whose name *starts*
    with ``tag_or_prefix`` (handy when only the lambda_GB part of the tag is
    known).
    """
    candidate = os.path.join(data_dir, tag_or_prefix)
    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "solution.npy")):
        return candidate
    for name in sorted(os.listdir(data_dir)):
        if name.startswith(tag_or_prefix):
            full = os.path.join(data_dir, name)
            if os.path.exists(os.path.join(full, "solution.npy")):
                return full
    raise FileNotFoundError(
        f"No run directory matching '{tag_or_prefix}' found under {data_dir}."
    )


# --------------------------------------------------------------------------- #
# Self-test / minimal example                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")              # batch-friendly default
    import matplotlib.pyplot as plt

    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--gr-tag",   default=DEFAULT_GR_TAG,   help="GR run tag (default: %(default)s)")
    cli.add_argument("--esgb-tag", default=DEFAULT_ESGB_TAG, help="EsGB run tag (default: %(default)s)")
    cli.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Run base directory (default: %(default)s)")
    cli.add_argument("--out-dir",  default=os.path.join(_PROJECT_ROOT, "RunningCode", "DATA", "GWfigs"),
                     help="Where to save figures (default: %(default)s)")
    cli.add_argument("--n-angles", type=int, default=20)
    cli.add_argument("--n-tau",    type=int, default=800)
    cli.add_argument("--N",        type=int, default=1, help="Number of oscillons")
    cli.add_argument("--Delta",    type=float, default=DEFAULT_DELTA)
    args = cli.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading runs ...")
    gr_run   = OscillonRun(find_run_dir(args.gr_tag,   args.data_dir), label="GR")
    esgb_run = OscillonRun(find_run_dir(args.esgb_tag, args.data_dir), label="EsGB")

    print("\n----- GR -----")
    gr_run.extract_all(verbose=True)
    print("\n----- EsGB -----")
    esgb_run.extract_all(verbose=True)

    # 0. Sanity-check the matter-domination assumption (w = 0)
    print("\n----- Equation-of-state sanity check -----")
    check_equation_of_state(gr_run)
    check_equation_of_state(esgb_run)

    # 1. Source power spectra
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    gr_run.plot_source_power_spectrum(ax=ax1, color="C0")
    esgb_run.plot_source_power_spectrum(ax=ax1, color="C3")
    fig1.tight_layout()
    p1 = os.path.join(args.out_dir, "source_power_spectrum.png")
    fig1.savefig(p1, dpi=140); print(f"  saved {p1}")

    # 2. GR vs EsGB Omega_GW with ratio panel
    fig2, _, (res_GR, res_EsGB) = compare_runs(
        gr_run, esgb_run,
        labels=("GR", "EsGB"),
        Delta=args.Delta, N=args.N,
        spec_kwargs=dict(n_angles=args.n_angles, n_tau=args.n_tau),
    )
    p2 = os.path.join(args.out_dir, "OmegaGW_GR_vs_EsGB.png")
    fig2.savefig(p2, dpi=140); print(f"  saved {p2}")

    # 3. Analytic A_GW(w), n_GW(w) -- sweep w explicitly since the GW
    #    pipeline assumes w = 0 by default; the sweep is purely to compare
    #    each run's amplitude/tilt against the analytic fit eqs. 4.18-4.21.
    w_sweep = np.linspace(-0.05, 0.40, 7)
    fig3, axes3, summary = plot_AGW_nGW_vs_w(
        [gr_run, esgb_run], Delta=args.Delta, N=args.N,
        w_values=w_sweep,
        spec_kwargs=dict(n_angles=args.n_angles, n_tau=args.n_tau),
    )
    p3 = os.path.join(args.out_dir, "AGW_nGW_vs_w.png")
    fig3.savefig(p3, dpi=140); print(f"  saved {p3}")

    print("\nDone.")
