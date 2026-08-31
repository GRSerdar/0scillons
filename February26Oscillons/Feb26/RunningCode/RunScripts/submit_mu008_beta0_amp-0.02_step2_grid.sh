#!/bin/bash -l
#
# A = -0.02 step-2 sub-grid filler at beta=0.
# Target full grid: lambda_GB in {0, 2, 4, 6, 8} x g2 in {0, 2, 4, ..., 20}
# plus the missing g2=0 endpoints at lambda_GB = 6 and 7.
#
# Already on disk or in flight (skipped automatically below):
#   lambda=0: full integer row 0..20
#   lambda=2: g2 in {0, 20}
#   lambda=4: g2 in {0, 4, 8, 12, 16, 20}
#   lambda=6: g2 in {20}
#   lambda=7: g2 in {20}
#   lambda=8: g2 in {0, 4, 8, 12, 16, 20}
#
# Fixed: mu=0.08, R=3, beta=0 (coupling=quadratic_0), a=b=0, dr=1/24, T=800.
# Idempotent: skips runs whose solution.npy already exists for the tag.

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

# Cells to submit: (lambda_GB, g2) pairs.
PAIRS=(
    "6 0"   "7 0"
    "2 2"  "2 4"  "2 6"  "2 8"  "2 10"  "2 12"  "2 14"  "2 16"  "2 18"
    "4 2"  "4 6"  "4 10" "4 14" "4 18"
    "6 2"  "6 4"  "6 6"  "6 8"  "6 10"  "6 12"  "6 14"  "6 16"  "6 18"
    "8 2"  "8 6"  "8 10" "8 14" "8 18"
)

submitted=0
skipped=0
errors=0

for pair in "${PAIRS[@]}"; do
    read -r LGB G2 <<< "${pair}"
    TAG="lgb${LGB}_mu${MU}_a${A_MG}_b${B_MG}_amp${AMP}_R${WIDTH}_dr0.0416667_${COUPLING}_g2${G2}"
    if [ -f "${OSC_DIR}/${TAG}/solution.npy" ]; then
        echo "  [skip] ${TAG} (solution.npy already exists)"
        skipped=$((skipped + 1))
        continue
    fi
    echo "  [send] lambda_GB=${LGB}, g2=${G2}, amp=${AMP}, coupling=${COUPLING}"
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
        echo "          (probably hitting a QOS submit limit; re-run later.)"
        errors=$((errors + 1))
    fi
done

echo ""
echo "Submitted: ${submitted}, Skipped: ${skipped}, Errors: ${errors}, Total in list: ${#PAIRS[@]}"
echo "Check queue with:  squeue -u \$USER --clusters=wice"
