#!/bin/bash -l
#
# Amplitude A = -0.1 repeat of the (lambda_GB, g2) scans at
# mu=0.08, R=3, beta=0 (coupling=quadratic_0), a=b=0, dr=1/24, T=800.
#
# Two combined scans:
#   1) g2  = 0,  lambda_GB = 0, 2, 4, 6, 8
#   2) lambda_GB = 0,  g2 = 0, 2, 4, 8, 10, 12
#
# The (lambda_GB=0, g2=0) cell is shared by both scans and is submitted once.
#
# Idempotent: skips runs whose solution.npy already exists for the new tag.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_output"

OSC_DIR="${VSC_DATA:-/data/leuven/384/vsc38419}/oscillon_runs"

AMP=-0.1
MU=0.08
WIDTH=3
A_MG=0
B_MG=0
COUPLING=quadratic_0
MIN_DR=0.041666666666666664   # 1/24
T=800

# (lambda_GB, g2) pairs, deduplicated.
PAIRS=(
    "0 0"
    "2 0"   "4 0"   "6 0"   "8 0"
    "0 2"   "0 4"   "0 8"   "0 10"  "0 12"
)

submitted=0
skipped=0

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
#SBATCH --job-name=l${LGB}_g${G2}_A01

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
    fi
done

echo ""
echo "Submitted: ${submitted}, Skipped: ${skipped}, Total in list: ${#PAIRS[@]}"
echo "Check queue with:  squeue -u \$USER --clusters=wice"
