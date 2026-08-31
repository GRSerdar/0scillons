#!/bin/bash -l
#
# Complete the lambda_GB=10 row of the (lgb, g2) compactness heatmap at
# beta=0, A=-0.02, R=3, mu=0.08, dr=1/24, T=800.
#
# Same fixed parameters as submit_mu008_beta0_amp-0.02_step2_grid.sh and
# submit_lgb10_g2_fill.sh.
#
# Targets the 6 cells still missing from the heatmap:
#   * lgb=10, g2=6                   -> previous run died at t=7.2 (target T=800); re-run.
#   * lgb=10, g2=12, 14, 16, 18, 20  -> never launched.
#
# Force-submits all six (no skip guard) so the broken g2=6 run is overwritten.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_output"

OSC_DIR="${VSC_DATA:-/data/leuven/384/vsc38419}/oscillon_runs"

AMP=-0.02
MU=0.08
WIDTH=3
A_MG=0
B_MG=0
COUPLING=quadratic_0
MIN_DR=0.041666666666666664   # 1/24
T=800

PAIRS=(
    "10 6"  "10 12"  "10 14"  "10 16"  "10 18"  "10 20"
)

submitted=0
errors=0

for pair in "${PAIRS[@]}"; do
    read -r LGB G2 <<< "${pair}"
    TAG="lgb${LGB}_mu${MU}_a${A_MG}_b${B_MG}_amp${AMP}_R${WIDTH}_dr0.0416667_${COUPLING}_g2${G2}"
    echo "  [send] lambda_GB=${LGB}, g2=${G2}, amp=${AMP}, coupling=${COUPLING} (force overwrite)"
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
#SBATCH --job-name=l${LGB}_g${G2}_A002

module purge
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0

echo "========================================================"
echo "  Job ID    : \${SLURM_JOB_ID}"
echo "  Node      : \$(hostname)"
echo "  lambda_GB : ${LGB}"
echo "  g2        : ${G2}"
echo "  amp       : ${AMP}"
echo "  width R   : ${WIDTH}"
echo "  mu        : ${MU}"
echo "  coupling  : ${COUPLING}"
echo "  a_mg, b_mg: ${A_MG}, ${B_MG}"
echo "  dr        : 1/24"
echo "  T         : ${T}"
echo "  Started   : \$(date)"
echo "========================================================"

python3 "${SCRIPT_DIR}/run_oscillon.py" \\
    --lambda_gb=${LGB} \\
    --selfinteraction=${MU} \\
    --perturbation=${AMP} \\
    --width=${WIDTH} \\
    --a_mg=${A_MG} \\
    --b_mg=${B_MG} \\
    --g2=${G2} \\
    --coupling=${COUPLING} \\
    --min_dr=${MIN_DR} \\
    --T=${T} \\
    --force

echo "Finished: \$(date)"
EOF
    rc=$?
    if [ $rc -eq 0 ]; then
        submitted=$((submitted + 1))
    else
        echo "  [error] sbatch failed for lgb=${LGB}, g2=${G2} (rc=${rc})"
        errors=$((errors + 1))
    fi
done

echo ""
echo "Submitted: ${submitted}, Errors: ${errors}, Total: ${#PAIRS[@]}"
echo "Check queue with:  squeue -u \$USER --clusters=wice"
