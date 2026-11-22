#!/bin/bash -l
#
# Rerun of the curves drawn in RunningCode/DATA/finalplots3.ipynb,
# with the corrected initial-data solver. Output folders get the
# suffix _idfix, so the existing solution.npy files are left in place.
#
# Plotted set:
#   beta = 0 (coupling quadratic_0), mu=0.08, A=-0.02, R=3, a=b=0
#     lambda_GB = 0, g2 = 0,2,4,6,8,10,12,14,16,18,20
#     g2 = 0, lambda_GB = 1,2,3,4,5
#   beta = 250 (coupling quadratic), same amplitude and width
#     lambda_GB = 0, g2 = 0, 6, 10, 34
#
# Grid matches the plotted runs: dr=1/24, r_max=150, T=800, chi0=0.15.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_output"

PYTHON="/groups/astro/syildiz/VSC_home/0scillons/.conda/oscillons/bin/python"
CONDA_LIB="/groups/astro/syildiz/VSC_home/0scillons/.conda/oscillons/lib"

AMP=-0.02
MU=0.08
WIDTH=3
A_MG=0
B_MG=0
MIN_DR=0.041666666666666664   # 1/24
T=800
SUFFIX=idfix

# coupling lambda_GB g2
PAIRS=(
    "quadratic_0 0 0"
    "quadratic_0 0 2"
    "quadratic_0 0 4"
    "quadratic_0 0 6"
    "quadratic_0 0 8"
    "quadratic_0 0 10"
    "quadratic_0 0 12"
    "quadratic_0 0 14"
    "quadratic_0 0 16"
    "quadratic_0 0 18"
    "quadratic_0 0 20"
    "quadratic_0 1 0"
    "quadratic_0 2 0"
    "quadratic_0 3 0"
    "quadratic_0 4 0"
    "quadratic_0 5 0"
    "quadratic 0 0"
    "quadratic 0 6"
    "quadratic 0 10"
    "quadratic 0 34"
)

submitted=0
errors=0

for pair in "${PAIRS[@]}"; do
    read -r COUPLING LGB G2 <<< "${pair}"
    echo "  [send] coupling=${COUPLING}, lambda_GB=${LGB}, g2=${G2}"
    sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=id_${COUPLING}_l${LGB}_g${G2}
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
echo "  suffix    : ${SUFFIX}"
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
    --T=${T} \\
    --tag-suffix=${SUFFIX}

echo "Finished: \$(date)"
EOF
    rc=$?
    if [ $rc -eq 0 ]; then
        submitted=$((submitted + 1))
    else
        echo "  [error] sbatch failed for coupling=${COUPLING}, lgb=${LGB}, g2=${G2} (rc=${rc})"
        errors=$((errors + 1))
    fi
done

echo ""
echo "Submitted: ${submitted}, Errors: ${errors}, Total in list: ${#PAIRS[@]}"
echo "Check queue with:  squeue -u \$USER -p astro2_long"
