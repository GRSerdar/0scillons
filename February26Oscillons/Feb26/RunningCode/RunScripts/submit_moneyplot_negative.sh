#!/bin/bash -l
#
# Negative-coupling counterparts of the compactness money plot
# (finalplots2.ipynb, panels 4b and 4c).
#
# Right panel (g2 scan at lambda_GB = 0):
#   g2 = 2, 4, 6, 8, 10, 12, 14, 16, 18, 20  ->  same values, negative.
# Left panel (GB scan at g2 = 0):
#   lambda_GB = 1, 2, 3, 4, 5  ->  same values, negative.
# g2 = 0 and lambda_GB = 0 are unchanged under a sign flip and are omitted.
#
# Same physics/grid as the plotted runs:
#   mu=0.08, A=-0.02, R=3, a=b=0, beta=0 (coupling=quadratic_0),
#   dr=1/24, r_max=150, T=800, chi0=0.15
#
# Tycho: account=astro, partition=astro2_long, conda env `oscillons`.
# Each pair is an independent 1-CPU job. Existing solution.npy is kept.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_output"

PYTHON="/groups/astro/syildiz/VSC_home/0scillons/.conda/oscillons/bin/python"
CONDA_LIB="/groups/astro/syildiz/VSC_home/0scillons/.conda/oscillons/lib"

AMP=-0.02
MU=0.08
WIDTH=3
A_MG=0
B_MG=0
COUPLING=quadratic_0
MIN_DR=0.041666666666666664   # 1/24
T=800

# (lambda_GB, g2)
PAIRS=(
    "0 -2" "0 -4" "0 -6" "0 -8" "0 -10"
    "0 -12" "0 -14" "0 -16" "0 -18" "0 -20"
    "-1 0" "-2 0" "-3 0" "-4 0" "-5 0"
)

submitted=0
errors=0

for pair in "${PAIRS[@]}"; do
    read -r LGB G2 <<< "${pair}"
    echo "  [send] lambda_GB=${LGB}, g2=${G2}"
    sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=l${LGB}_g${G2}_neg
#SBATCH --account=astro
#SBATCH --partition=astro2_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=${SCRIPT_DIR}/slurm_output/%j.txt
#SBATCH --error=${SCRIPT_DIR}/slurm_output/%j.err

export PATH="${PYTHON%/*}:\${PATH}"
export LD_LIBRARY_PATH="${CONDA_LIB}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

echo "========================================================"
echo "  Job ID    : \${SLURM_JOB_ID}"
echo "  Node      : \$(hostname)"
echo "  Python    : ${PYTHON}"
echo "  lambda_GB : ${LGB}"
echo "  g2        : ${G2}"
echo "  coupling  : ${COUPLING}"
echo "  a_mg, b_mg: ${A_MG}, ${B_MG}"
echo "  dr        : 1/24"
echo "  T         : ${T}"
echo "  Started   : \$(date)"
echo "========================================================"

${PYTHON} -c "import numpy,scipy,tqdm; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

${PYTHON} "${SCRIPT_DIR}/run_oscillon.py" \\
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
        echo "  [error] sbatch failed for lgb=${LGB}, g2=${G2} (rc=${rc})"
        errors=$((errors + 1))
    fi
done

echo ""
echo "Submitted: ${submitted}, Errors: ${errors}, Total in list: ${#PAIRS[@]}"
echo "Check queue with:  squeue -u \$USER -p astro2_long"
