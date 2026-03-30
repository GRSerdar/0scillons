#!/bin/bash -l
#
# Submit one gauge-testing oscillon simulation to SLURM (VSC Genius).
#
# Usage:
#   sbatch submit_gauge_test.sh                          # all defaults
#   sbatch submit_gauge_test.sh 5.0                      # positional lambda_GB
#   GAUGE=baumgarte MIN_DR=0.03125 sbatch submit_gauge_test.sh 5.0
#
# All parameters can be overridden via environment variables:
#   LGB, MU, T, NPTS, COUPLING, CHI0, AMP, WIDTH,
#   A_MG, B_MG, GAUGE, ETA, MIN_DR, MAX_DR, R_MAX, FORCE
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
LAMBDA_GB="${LGB:-${1:-5.0}}"
SELFINTERACTION="${MU:-0.08}"
T_EVOL="${T:-800}"
NPTS_VAL="${NPTS:-1000}"
COUPLING_TYPE="${COUPLING:-quadratic}"
CHI0_VAL="${CHI0:-0.15}"
PERTURB="${AMP:--2e-2}"
WIDTH_VAL="${WIDTH:-3}"

# Gauge parameters
GAUGE_TYPE="${GAUGE:-modified_harmonic}"
ETA_VAL="${ETA:-1.0}"
A_MG_VAL="${A_MG:-0.2}"
B_MG_VAL="${B_MG:-0.4}"

# Resolution parameters
MIN_DR_VAL="${MIN_DR:-0.0625}"
MAX_DR_VAL="${MAX_DR:-2}"
R_MAX_VAL="${R_MAX:-150}"
SPACING_VAL="${SPACING:-cubic}"
SINH_A_VAL="${SINH_A:-}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${SCRIPT_DIR}/slurm_output"

echo "========================================================"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : $(hostname)"
echo "  gauge_type  : ${GAUGE_TYPE}"
echo "  eta         : ${ETA_VAL}"
echo "  a_mg, b_mg  : ${A_MG_VAL}, ${B_MG_VAL}"
echo "  lambda_GB   : ${LAMBDA_GB}"
echo "  mu          : ${SELFINTERACTION}"
echo "  coupling    : ${COUPLING_TYPE}"
echo "  amplitude   : ${PERTURB}"
echo "  width       : ${WIDTH_VAL}"
echo "  T           : ${T_EVOL}"
echo "  min_dr      : ${MIN_DR_VAL}"
echo "  max_dr      : ${MAX_DR_VAL}"
echo "  r_max       : ${R_MAX_VAL}"
echo "  spacing     : ${SPACING_VAL}"
echo "  sinh_a      : ${SINH_A_VAL:-auto}"
echo "  Started     : $(date)"
echo "========================================================"

FORCE_FLAG="${FORCE:+--force}"
SINH_A_FLAG=""
if [ -n "${SINH_A_VAL}" ]; then
    SINH_A_FLAG="--sinh_a=${SINH_A_VAL}"
fi

python3 "${SCRIPT_DIR}/run_gauge_test.py" \
    --gauge_type="${GAUGE_TYPE}" \
    --eta="${ETA_VAL}" \
    --lambda_gb="${LAMBDA_GB}" \
    --selfinteraction="${SELFINTERACTION}" \
    --a_mg="${A_MG_VAL}" \
    --b_mg="${B_MG_VAL}" \
    --chi0="${CHI0_VAL}" \
    --coupling="${COUPLING_TYPE}" \
    --perturbation="${PERTURB}" \
    --width="${WIDTH_VAL}" \
    --T="${T_EVOL}" \
    --num_points_t="${NPTS_VAL}" \
    --r_max="${R_MAX_VAL}" \
    --min_dr="${MIN_DR_VAL}" \
    --max_dr="${MAX_DR_VAL}" \
    --spacing="${SPACING_VAL}" \
    ${SINH_A_FLAG} \
    ${FORCE_FLAG}

echo "Finished: $(date)"
