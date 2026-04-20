#!/bin/bash -l
#
# Resubmit g2 scan after build_run_tag fix (g2 now included in dir name).
# lambda_GB in {0, 1}, g2 in {0.001, 0.01, 0.1}, a=b=0, T=800
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_output"

for LGB in 0 1; do
    for G2 in 0.001 0.01 0.1; do
        echo "Submitting lambda_GB=${LGB}, g2=${G2} ..."
        sbatch <<EOF
#!/bin/bash -l
#SBATCH --output=${SCRIPT_DIR}/slurm_output/%j.txt
#SBATCH -e ${SCRIPT_DIR}/slurm_output/%j.err
#SBATCH --account=intro_vsc38419
#SBATCH --partition=batch_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --cluster=genius
#SBATCH --job-name=l${LGB}_g${G2}

module purge
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0

echo "========================================================"
echo "  Job ID    : \${SLURM_JOB_ID}"
echo "  Node      : \$(hostname)"
echo "  lambda_GB : ${LGB}"
echo "  g2        : ${G2}"
echo "  a_mg, b_mg: 0, 0"
echo "  T         : 800"
echo "  Started   : \$(date)"
echo "========================================================"

python3 "${SCRIPT_DIR}/run_oscillon.py" \
    --lambda_gb=${LGB} \
    --a_mg=0 \
    --b_mg=0 \
    --g2=${G2} \
    --T=800 \
    --force

echo "Finished: \$(date)"
EOF
    done
done
