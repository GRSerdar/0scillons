#!/bin/bash -l
#
# Fill in the missing (lambda_GB, g2) cells of the Panel 4a heatmap.
# Fixed parameters: mu=0.08, A=-0.02, R=3, a=b=0, beta=0 (coupling=quadratic_0),
# dr=1/24, T=800.  Same convention as submit_g2_lambda_scan.sh so the loader
# in FinalPlots.ipynb assigns beta=0 and the right g2 to each run's key.
#
# Missing entirely (30 runs):
#   lgb=  0: g2 = 8, 12, 16
#   lgb=  4: g2 = 4, 8, 12, 16
#   lgb=  8: g2 = 0, 4, 8, 12, 16
#   lgb= 12: g2 = 0, 4, 8, 12, 16, 20
#   lgb= 16: g2 = 0, 4, 8, 12, 16, 20
#   lgb= 20: g2 = 0, 4, 8, 12, 16, 20
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_output"

# (lgb, g2) pairs to submit
PAIRS=(
    "0 8"   "0 12"  "0 16"
    "4 4"   "4 8"   "4 12"  "4 16"
    "8 0"   "8 4"   "8 8"   "8 12"  "8 16"
    "12 0"  "12 4"  "12 8"  "12 12" "12 16" "12 20"
    "16 0"  "16 4"  "16 8"  "16 12" "16 16" "16 20"
    "20 0"  "20 4"  "20 8"  "20 12" "20 16" "20 20"
)

echo "Submitting ${#PAIRS[@]} jobs to wICE (batch)..."

for pair in "${PAIRS[@]}"; do
    read -r LGB G2 <<< "${pair}"
    echo "  lambda_GB=${LGB}, g2=${G2}, coupling=quadratic_0"
    sbatch <<EOF
#!/bin/bash -l
#SBATCH --output=${SCRIPT_DIR}/slurm_output/%j.txt
#SBATCH -e ${SCRIPT_DIR}/slurm_output/%j.err
#SBATCH --account=lp_nr
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --clusters=wice
#SBATCH --job-name=l${LGB}_g${G2}

module purge
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0

echo "========================================================"
echo "  Job ID    : \${SLURM_JOB_ID}"
echo "  Node      : \$(hostname)"
echo "  lambda_GB : ${LGB}"
echo "  g2        : ${G2}"
echo "  coupling  : quadratic_0"
echo "  a_mg, b_mg: 0, 0"
echo "  dr        : 1/24"
echo "  T         : 800"
echo "  Started   : \$(date)"
echo "========================================================"

python3 "${SCRIPT_DIR}/run_oscillon.py" \\
    --lambda_gb=${LGB} \\
    --selfinteraction=0.08 \\
    --perturbation=-0.02 \\
    --width=3 \\
    --a_mg=0 \\
    --b_mg=0 \\
    --g2=${G2} \\
    --coupling=quadratic_0 \\
    --min_dr=0.041666666666666664 \\
    --T=800 \\
    --force

echo "Finished: \$(date)"
EOF
done

echo ""
echo "All jobs submitted.  Check queue with:  squeue -u \$USER --clusters=wice"
