#!/bin/bash -l
#
# Convergence test at lambda_GB = 0, g2 = 10 (wider spread, factor-of-2
# refinement between adjacent pairs).
# Three resolutions:
#   LR : min_dr = 1/12 (~0.08333)  -- submitted here.
#   MR : min_dr = 1/24 (~0.04167)  -- already exists on disk, skipped.
#   HR : min_dr = 1/48 (~0.02083)  -- submitted here.
# All other parameters match the standard "quadratic_0" g2-grid runs
# (no rampOn): mu=0.08, amp=-0.02, width=3, a_mg=0, b_mg=0, T=800.
#
# Mirrors submit_lgb5_g25_convergence.sh, so the compactness convergence
# cell in DATA/compactness.ipynb can load the three runs the same way.
# NOTE: previous version of this script used dr in (1/18, 1/24, 1/30);
# those runs are still on disk if you need them for cross-checks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_output"

OSC_DIR="${VSC_DATA:-/data/leuven/384/vsc38419}/oscillon_runs"

LGB=0
G2=10

AMP=-0.02
MU=0.08
WIDTH=3
A_MG=0
B_MG=0
COUPLING=quadratic_0
T=800

# (min_dr, dr_tag_for_filename, walltime, jobname_suffix)
# Walltimes are ~2-3x the empirical runtime observed on this parameter combo
# (LR ≈ 1 h, MR ≈ 3 h, HR ≈ 10 h) so any transient node load can't kill us.
CONFIGS=(
    "0.08333333333333333 0.0833333 04:00:00 r12"
    "0.04166666666666666 0.0416667 08:00:00 r24"
    "0.02083333333333333 0.0208333 24:00:00 r48"
)

submitted=0
skipped=0

for cfg in "${CONFIGS[@]}"; do
    read -r MIN_DR DR_TAG WALLTIME SUFFIX <<< "${cfg}"
    TAG="lgb${LGB}_mu${MU}_a${A_MG}_b${B_MG}_amp${AMP}_R${WIDTH}_dr${DR_TAG}_${COUPLING}_g2${G2}"
    if [ -f "${OSC_DIR}/${TAG}/solution.npy" ]; then
        echo "  [skip] ${TAG} (solution.npy already exists)"
        skipped=$((skipped + 1))
        continue
    fi
    echo "  [send] lambda_GB=${LGB}, g2=${G2}, min_dr=${MIN_DR}, walltime=${WALLTIME}"
    sbatch <<EOF
#!/bin/bash -l
#SBATCH --output=${SCRIPT_DIR}/slurm_output/%j.txt
#SBATCH -e ${SCRIPT_DIR}/slurm_output/%j.err
#SBATCH --account=lp_nr
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=${WALLTIME}
#SBATCH --clusters=wice
#SBATCH --job-name=l${LGB}_g${G2}_${SUFFIX}

module purge
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0

echo "========================================================"
echo "  Job ID    : \${SLURM_JOB_ID}"
echo "  Node      : \$(hostname)"
echo "  lambda_GB : ${LGB}"
echo "  g2        : ${G2}"
echo "  min_dr    : ${MIN_DR}"
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
    --T=${T}

echo "Finished: \$(date)"
EOF
    rc=$?
    if [ $rc -eq 0 ]; then
        submitted=$((submitted + 1))
    else
        echo "  [error] sbatch failed (rc=${rc})"
    fi
done

echo ""
echo "Submitted: ${submitted}, Skipped: ${skipped}, Total: ${#CONFIGS[@]}"
echo "Check queue with:  squeue -u \$USER --clusters=wice"
