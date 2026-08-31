#!/bin/bash -l
#
# Follow-up to submit_g2_lgb_grid_missing.sh.
# The first batch hit the wICE `long` QoS submit limit of 20.
# Re-run this script as soon as slots free up.
#
# It is idempotent:
#   - if a run's solution.npy already exists, the run is skipped.
#   - otherwise it is (re-)submitted with --force.
#
# Same fixed parameters as the first batch:
#   mu=0.08, A=-0.02, R=3, a=b=0, beta=0 (coupling=quadratic_0),
#   dr=1/24, T=800.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_output"

OSC_DIR="${VSC_DATA:-/data/leuven/384/vsc38419}/oscillon_runs"

# Remaining (lgb, g2) cells of the heatmap.
PAIRS=(
    "12 20"
    "16 0"   "16 4"   "16 8"   "16 12"  "16 16"  "16 20"
    "20 0"   "20 4"   "20 8"   "20 12"  "20 16"  "20 20"
)

submitted=0
skipped=0

for pair in "${PAIRS[@]}"; do
    read -r LGB G2 <<< "${pair}"
    TAG="lgb${LGB}_mu0.08_a0_b0_amp-0.02_R3_dr0.0416667_quadratic_0_g2${G2}"
    if [ -f "${OSC_DIR}/${TAG}/solution.npy" ]; then
        echo "  [skip] ${TAG} (solution.npy already exists)"
        skipped=$((skipped + 1))
        continue
    fi
    echo "  [send] lambda_GB=${LGB}, g2=${G2}, coupling=quadratic_0"
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
    rc=$?
    if [ $rc -eq 0 ]; then
        submitted=$((submitted + 1))
    else
        echo "  [error] sbatch failed for lgb=${LGB}, g2=${G2} (rc=${rc})"
        echo "          (probably still hitting the QOSMaxSubmitJobPerUserLimit; try again later.)"
    fi
done

echo ""
echo "Submitted: ${submitted}, Skipped: ${skipped}, Total in list: ${#PAIRS[@]}"
echo "Check queue with:  squeue -u \$USER --clusters=wice"
