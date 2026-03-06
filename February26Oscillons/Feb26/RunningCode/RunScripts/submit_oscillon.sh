#!/bin/bash -l
#
# Submit one oscillon simulation to the SLURM scheduler.
#
# Usage:
#   sbatch submit_oscillon.sh                     # default: lambda_GB = 0.03
#   sbatch submit_oscillon.sh 0.05                # custom lambda_GB
#   LGB=0.05 T=1200 sbatch submit_oscillon.sh     # override via env vars
#

#SBATCH -J oscMG
#SBATCH --output=slurm_output/%j.txt
#SBATCH -e slurm_output/%j.err
#SBATCH --account=ns-users
#SBATCH --partition=ns-main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=14-00:00:00

# ── Parameters (override with env vars or positional arg) ────────────────────
LAMBDA_GB="${LGB:-${1:-0.03}}"
SELFINTERACTION="${MU:-0.08}"
T_EVOL="${T:-800}"
NPTS="${NPTS:-1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Node      : $(hostname)"
echo "  lambda_GB : ${LAMBDA_GB}"
echo "  T         : ${T_EVOL}"
echo "  Started   : $(date)"
echo "========================================================"

python3 "${SCRIPT_DIR}/run_oscillon.py" \
    --lambda_gb "${LAMBDA_GB}" \
    --selfinteraction "${SELFINTERACTION}" \
    --T "${T_EVOL}" \
    --num_points_t "${NPTS}"

echo "Finished: $(date)"
