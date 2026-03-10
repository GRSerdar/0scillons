#!/bin/bash -l
#
# Submit one oscillon simulation to the SLURM scheduler (VSC Genius).
#
# Usage:
#   sbatch submit_oscillon.sh                     # default: lambda_GB = 0.03
#   sbatch submit_oscillon.sh 0.05                # custom lambda_GB
#   LGB=0.05 T=1200 sbatch submit_oscillon.sh     # override via env vars
#

#SBATCH --output=slurm_output/%j.txt
#SBATCH -e slurm_output/%j.err
#SBATCH --account=intro_vsc38419
#SBATCH --partition=batch_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --cluster=genius

# ── Modules ──────────────────────────────────────────────────────────────────
module purge
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0

# ── Parameters (override with env vars or positional arg) ────────────────────
LAMBDA_GB="${LGB:-${1:-0.03}}"
SELFINTERACTION="${MU:-0.08}"
T_EVOL="${T:-800}"
NPTS="${NPTS:-1000}"
COUPLING_TYPE="${COUPLING:-quadratic}"
CHI0_VAL="${CHI0:-0.15}"
PERTURB="${AMP:--2e-2}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${SCRIPT_DIR}/slurm_output"

echo "========================================================"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Node      : $(hostname)"
echo "  lambda_GB : ${LAMBDA_GB}"
echo "  T         : ${T_EVOL}"
echo "  Started   : $(date)"
echo "========================================================"

FORCE_FLAG="${FORCE:+--force}"

python3 "${SCRIPT_DIR}/run_oscillon.py" \
    --lambda_gb="${LAMBDA_GB}" \
    --selfinteraction="${SELFINTERACTION}" \
    --T="${T_EVOL}" \
    --num_points_t="${NPTS}" \
    --coupling="${COUPLING_TYPE}" \
    --chi0="${CHI0_VAL}" \
    --perturbation="${PERTURB}" \
    ${FORCE_FLAG}

echo "Finished: $(date)"
