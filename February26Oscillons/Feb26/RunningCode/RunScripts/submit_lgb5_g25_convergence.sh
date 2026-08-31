#!/bin/bash -l
#
# Convergence test at lambda_GB = 5, g2 = 5.
# Two resolutions:  min_dr = 1/18 (~0.05556)  and  min_dr = 1/30 (~0.03333).
# All other parameters match the standard "quadratic_0" g2-grid runs
# (no rampOn): mu=0.08, amp=-0.02, width=3, a_mg=0, b_mg=0, T=800.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_output"

OSC_DIR="${VSC_DATA:-/data/leuven/384/vsc38419}/oscillon_runs"

LGB=5
G2=5

AMP=-0.02
MU=0.08
WIDTH=3
A_MG=0
B_MG=0
COUPLING=quadratic_0
T=800

# (min_dr, dr_tag_for_filename, walltime, jobname_suffix)
CONFIGS=(
    "0.05555555555555555 0.0555556 2-00:00:00 r18"
    "0.03333333333333333 0.0333333 3-00:00:00 r30"
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
