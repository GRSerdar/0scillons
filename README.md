# Oscillons during Reheating

Spherically symmetric numerical relativity code for oscillons during reheating,
in both GR and Einstein-scalar-Gauss-Bonnet modified gravity.

## Layout

```
February26Oscillons/Feb26/
├── core/            grid, spacing, derivatives, state vector, RHS evolution
├── bssn/            BSSN variables/RHS, modified gravity, diagnostics, GW signal
├── matter/          scalar matter (with modified-gravity coupling)
├── initialdata/     oscillaton and modified-gravity initial conditions
├── backgrounds/     flat spherical background
├── Notebooks/       exploratory and convergence notebooks
└── RunningCode/
    ├── RunScripts/    run_oscillon.py + SLURM submission scripts
    ├── GaugeTesting/  gauge comparison runs and notebook
    └── DATA/          analysis notebooks and final figures (GoodFigures/)
tools/               helper scripts (run manifest)
run_manifest.csv     parameters and outcome of every completed run
```

## Environment

Pure Python, no compiled extensions. Dependencies are `numpy`, `scipy`,
`matplotlib`, `mpmath`, `tqdm` (see `requirements.txt`).

On a fresh machine:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

On the KU Leuven (Genius/wICE) clusters the submission scripts instead relied on
modules, which avoids building SciPy:

```bash
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
```

Those modules provide Python 3.12.3, numpy 1.26.4, scipy 1.13.1, mpmath 1.3.0
and tqdm 4.66.5. They do **not** provide matplotlib — on the cluster it came
from a `pip install --user` into `~/.local`, so it has to be installed
separately. `requirements-pinned.txt` records the exact versions the archived
runs were produced with; use it when reproducing published numbers, and
`requirements.txt` for ordinary development.

On any other SLURM cluster, either load the equivalent modules or point the
submission scripts at a virtualenv.

## Where simulation output goes

`run_oscillon.py` writes each run to:

```
$VSC_DATA/oscillon_runs/<run_tag>/
```

and falls back to `RunningCode/DATA/oscillon_runs/<run_tag>/` when `VSC_DATA` is
unset. On a new cluster, set `VSC_DATA` to a large scratch/data filesystem —
the full run set is roughly 13 GB across ~360 runs.

`run_tag` encodes the parameters, e.g.

```
lgb10_mu0.08_a0_b0_amp-0.02_R3_dr0.0416667_quadratic_0_g214
```

Each run directory contains:

| File | Contents | Typical size |
| --- | --- | --- |
| `solution.npy` | full state vector on the (t, r) grid | ~24 MB |
| `t.npy`, `r.npy` | output times and grid radii | a few kB |
| `metadata.npz` | all run parameters, wall time, solver status | ~5 kB |
| `diagnostics.npz` | oscillon diagnostics (compactness, mass, radius, ...) | ~5 MB |
| `eft_diagnostics.npz` | EFT validity diagnostics, where computed | ~40 MB |

The output data is **not** in this repository. `oscillon_runs_data` at the repo
root is a symlink to the KU Leuven data filesystem and will dangle elsewhere;
repoint or delete it as needed.

## Running a simulation

Directly:

```bash
cd February26Oscillons/Feb26/RunningCode/RunScripts
python3 run_oscillon.py --lambda_gb 10 --g2 14 --coupling quadratic_0 \
    --selfinteraction 0.08 --perturbation -0.02 --width 3 \
    --a_mg 0 --b_mg 0 --min_dr 0.041666666666666664 --T 800
```

See `python3 run_oscillon.py --help` for all parameters. Runs skip themselves if
`solution.npy` already exists unless `--force` is given.

Via SLURM, the `submit_*.sh` scripts in `RunScripts/` each submit one parameter
sweep and already skip runs present on disk. They hardcode
`--account=lp_nr --clusters=wice`, so those lines need changing on a new
cluster. `submit_lgb0_g210_convergence.sh` is a good short template.

## Analysis

The notebooks in `RunningCode/DATA/` load runs straight out of
`$VSC_DATA/oscillon_runs`. `finalplots2.ipynb` and `compactness.ipynb` produce
the final figures, which are committed under `RunningCode/DATA/GoodFigures/`.

## Run manifest

`run_manifest.csv` lists every run that was completed, with its parameters,
solver outcome, wall time, and which output files it has. Regenerate it with:

```bash
python3 tools/make_run_manifest.py "$VSC_DATA/oscillon_runs" -o run_manifest.csv
```

Use it to see which parameter combinations exist without needing the data
itself, and to reproduce any subset of runs on new hardware.
