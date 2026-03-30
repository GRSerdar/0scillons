#!/usr/bin/env python3
"""
Gauge-testing oscillon simulation in Einstein-scalar-Gauss-Bonnet gravity.

Like run_oscillon.py but designed for comparing gauge conditions and
resolutions.  Output goes to gauge_testing/<run_tag>/ so it never
interferes with the production runs in oscillon_runs/.

Usage:
    python run_gauge_test.py --lambda_gb 5 --gauge_type modified_harmonic --min_dr 0.0625
    python run_gauge_test.py --lambda_gb 5 --gauge_type baumgarte --min_dr 0.03125
"""

import sys
import os
import gc
import argparse
import time
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.grid import Grid
from core.rhsevolution_MG import get_rhs
from core.spacing import CubicSpacing, SinhSpacing
from core.statevector import StateVector
from matter.scalarmatter_MG import ScalarMatter
from bssn.oscillondiagnostic import get_oscillon_diagnostic
from initialdata.ModifiedGravityInitialConditions import get_initial_state
from backgrounds.sphericalbackground import FlatSphericalBackground


TABLE_I = {
    0.04: dict(phi=-6.03334e-2, dphi=2.20256e-2, H=1.55744e-2),
    0.05: dict(phi=-7.36454e-2, dphi=2.72499e-2, H=1.92686e-2),
    0.06: dict(phi=-8.64102e-2, dphi=3.23761e-2, H=2.28934e-2),
    0.07: dict(phi=-9.86792e-2, dphi=3.74094e-2, H=2.64525e-2),
    0.08: dict(phi=-1.10495e-1, dphi=4.23540e-2, H=2.99488e-2),
    0.09: dict(phi=-1.21893e-1, dphi=4.72136e-2, H=3.33851e-2),
    0.10: dict(phi=-1.32906e-1, dphi=5.19917e-2, H=3.67637e-2),
}


def build_run_tag(gauge_type, eta, lgb, selfinteraction, a_mg, b_mg,
                  perturbation, width, min_dr, max_dr, coupling="quadratic",
                  spacing_type="cubic", sinh_a=None):
    tag = (
        f"gauge_{gauge_type}_eta{eta}"
        f"_lgb{lgb}_mu{selfinteraction}_a{a_mg}_b{b_mg}"
        f"_amp{perturbation}_R{width}"
        f"_dr{min_dr}_maxdr{max_dr}"
    )
    if spacing_type != "cubic":
        tag += f"_{spacing_type}"
    if sinh_a is not None:
        tag += f"_sinha{sinh_a}"
    if coupling != "quadratic":
        tag += f"_{coupling}"
    return tag


def run_simulation(args):
    """Set up the grid, initial data, evolve, and save."""

    scalar_mu = 1
    selfinteraction = args.selfinteraction

    entry = TABLE_I[selfinteraction]
    u_val = entry["phi"]
    v_val = entry["dphi"]

    perturbation = args.perturbation
    width = args.width
    chi0 = args.chi0
    coupling = args.coupling

    a_mg = args.a_mg
    b_mg = args.b_mg
    gauge_type = args.gauge_type
    eta = args.eta
    spacing_type = args.spacing
    sinh_a = args.sinh_a

    T = args.T
    num_points_t = args.num_points_t
    dt = T / num_points_t
    t_out = np.linspace(0, T - dt, num_points_t)

    r_max = args.r_max
    min_dr = args.min_dr
    max_dr = args.max_dr
    lgb = args.lambda_gb

    tag = build_run_tag(gauge_type, eta, lgb, selfinteraction, a_mg, b_mg,
                        perturbation, width, min_dr, max_dr, coupling, spacing_type,
                        sinh_a)
    vsc_data = os.environ.get("VSC_DATA", os.path.join(SCRIPT_DIR, "..", "DATA"))
    data_dir = os.path.join(vsc_data, "gauge_testing", tag)
    os.makedirs(data_dir, exist_ok=True)

    sol_path = os.path.join(data_dir, "solution.npy")
    if os.path.exists(sol_path) and not args.force:
        print(f"Data already exists at {data_dir}, use --force to overwrite.")
        return

    print("=" * 70)
    print(f"Gauge-testing oscillon simulation")
    print(f"  gauge_type    = {gauge_type}")
    print(f"  eta           = {eta}")
    print(f"  lambda_GB     = {lgb}")
    print(f"  selfinteraction (mu) = {selfinteraction}")
    print(f"  gauge (a, b)  = ({a_mg}, {b_mg})")
    print(f"  perturbation  = {perturbation},  width = {width}")
    print(f"  coupling      = {coupling}")
    print(f"  T = {T},  output points = {num_points_t}")
    sinh_a_str = f", sinh_a = {sinh_a}" if sinh_a is not None else ""
    print(f"  Grid: r_max = {r_max}, dr in [{min_dr}, {max_dr}], spacing = {spacing_type}{sinh_a_str}")
    print(f"  Output dir: {data_dir}")
    print("=" * 70)

    matter = ScalarMatter(scalar_mu, selfinteraction)
    sv = StateVector(matter)
    if spacing_type == "sinh":
        if sinh_a is not None:
            auto_params = SinhSpacing.get_parameters(r_max, min_dr, max_dr)
            num_points = auto_params["num_points"]
            spacing = SinhSpacing(num_points=num_points, r_max=r_max, a=sinh_a)
            print(f"  SinhSpacing: a={sinh_a} (override), auto_a={auto_params['a']:.4f}, "
                  f"num_points={num_points}, actual min_dr={spacing.min_dr:.6f}")
        else:
            spacing = SinhSpacing(**SinhSpacing.get_parameters(r_max, min_dr, max_dr))
    else:
        spacing = CubicSpacing(**CubicSpacing.get_parameters(r_max, min_dr, max_dr))
    grid = Grid(spacing, sv)
    bg = FlatSphericalBackground(grid.r)

    actual_min_dr = spacing.min_dr
    print(f"Grid: {grid.r.size} points,  r in [{grid.r[0]:.4f}, {grid.r[-1]:.1f}]")
    print(f"  actual min_dr = {actual_min_dr:.6f},  max_step = {0.4 * actual_min_dr:.6f}")

    params = (lgb, a_mg, b_mg, chi0, coupling)
    initial_state = get_initial_state(
        grid, bg, params, matter, perturbation, width, scalar_mu, u_val, v_val
    )

    print(f"\nStarting integration (T = {T}) ...")
    wall_start = time.time()

    with tqdm(total=1000, unit="\u2030") as pbar:
        dense_sol = solve_ivp(
            get_rhs,
            [0, T],
            initial_state,
            args=(grid, bg, matter, pbar, [0, T / 1000],
                  a_mg, b_mg, lgb, coupling, gauge_type, eta),
            max_step=0.4 * actual_min_dr,
            method="RK45",
            dense_output=True,
        )

    wall_elapsed = time.time() - wall_start
    crashed = not dense_sol.success
    t_crash = float(dense_sol.t[-1]) if crashed else T

    print(f"Integration finished in {wall_elapsed / 3600:.2f} h  "
          f"(status: {dense_sol.message})")

    if crashed:
        print(f"WARNING: solver failed at t ≈ {t_crash:.4f} — {dense_sol.message}")
        print(f"  Saving partial data up to t = {t_crash:.4f}")

    if crashed:
        t_out = t_out[t_out <= t_crash]
        if len(t_out) == 0:
            t_out = np.array([0.0])
    solution = dense_sol.sol(t_out).T

    np.save(sol_path, solution)
    np.save(os.path.join(data_dir, "t.npy"), t_out)
    np.save(os.path.join(data_dir, "r.npy"), grid.r)
    print(f"Saved solution: shape {solution.shape}  ->  {data_dir}")

    meta = dict(
        gauge_type=gauge_type, eta=eta, spacing_type=spacing_type,
        sinh_a=sinh_a if sinh_a is not None else -1.0,
        lambda_gb=lgb, selfinteraction=selfinteraction,
        a_mg=a_mg, b_mg=b_mg, chi0=chi0, coupling=coupling,
        perturbation=perturbation, width=width,
        T=T, num_points_t=num_points_t,
        r_max=r_max, min_dr=min_dr, max_dr=max_dr,
        wall_time_s=wall_elapsed,
        solver_message=dense_sol.message,
        solver_success=dense_sol.success,
        t_crash=t_crash if crashed else -1.0,
        num_grid_points=int(grid.r.size),
    )
    np.savez(os.path.join(data_dir, "metadata.npz"), **meta)

    if args.diagnostics:
        print("Computing oscillon diagnostics ...")
        osc = get_oscillon_diagnostic(
            solution, t_out, grid, bg,
            ScalarMatter(scalar_mu, selfinteraction),
            params,
            surface_threshold=0.05,
            r_max_diag=100.0,
        )
        np.savez(os.path.join(data_dir, "diagnostics.npz"), **osc)
        print(f"  max compactness C = {np.max(osc['C']):.6e}")

    del dense_sol, solution, initial_state, matter, sv, spacing, grid, bg
    gc.collect()
    print("Done.\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Gauge-testing oscillon simulation in EsGB gravity."
    )
    # ── Gauge parameters ──
    p.add_argument("--gauge_type", type=str, default="modified_harmonic",
                   choices=["modified_harmonic", "baumgarte", "bona_masso", "frozen"],
                   help="Gauge condition (default: modified_harmonic)")
    p.add_argument("--eta", type=float, default=1.0,
                   help="1+log damping coefficient eta (default: 1.0)")
    # ── Physics parameters ──
    p.add_argument("--lambda_gb", type=float, required=True,
                   help="Gauss-Bonnet coupling constant (0 = GR)")
    p.add_argument("--selfinteraction", type=float, default=0.08,
                   help="Self-interaction parameter mu (default: 0.08)")
    p.add_argument("--a_mg", type=float, default=0.2,
                   help="Modified gauge parameter a (default: 0.2)")
    p.add_argument("--b_mg", type=float, default=0.4,
                   help="Modified gauge parameter b (default: 0.4)")
    p.add_argument("--chi0", type=float, default=0.15,
                   help="chi0 parameter (default: 0.15)")
    p.add_argument("--coupling", type=str, default="quadratic",
                   help="Coupling type (default: quadratic)")
    p.add_argument("--perturbation", type=float, default=-2e-2,
                   help="Perturbation amplitude (default: -2e-2)")
    p.add_argument("--width", type=float, default=3,
                   help="Perturbation width R (default: 3)")
    # ── Time parameters ──
    p.add_argument("--T", type=float, default=800,
                   help="Total evolution time (default: 800)")
    p.add_argument("--num_points_t", type=int, default=1000,
                   help="Number of output time points (default: 1000)")
    # ── Grid / resolution parameters ──
    p.add_argument("--r_max", type=float, default=150,
                   help="Outer radius (default: 150)")
    p.add_argument("--min_dr", type=float, default=1/16,
                   help="Minimum grid spacing (default: 1/16 = 0.0625)")
    p.add_argument("--max_dr", type=float, default=2,
                   help="Maximum grid spacing (default: 2)")
    p.add_argument("--spacing", type=str, default="cubic",
                   choices=["cubic", "sinh"],
                   help="Grid spacing type (default: cubic)")
    p.add_argument("--sinh_a", type=float, default=None,
                   help="Override sinh spacing parameter a (default: auto from min_dr/max_dr)")
    # ── Misc ──
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing data")
    p.add_argument("--no-diagnostics", dest="diagnostics", action="store_false",
                   help="Skip oscillon diagnostics after evolution")
    p.set_defaults(diagnostics=True)
    return p.parse_args()


if __name__ == "__main__":
    run_simulation(parse_args())
