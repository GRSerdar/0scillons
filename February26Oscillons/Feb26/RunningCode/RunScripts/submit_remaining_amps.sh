#!/bin/bash -l
#
# Resubmit: amp=-0.05 lgb=0, amp=-0.1 lgb=5-8, amp=-0.5 lgb=0-8, amp=-1 lgb=0-8
# All: mu=0.08, quadratic beta=0, a=b=0, min_dr=1/24, T=800
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p slurm_output

MU=0.08
T=800
COUPLING=quadratic_0
A_MG=0
B_MG=0
MIN_DR=0.041666666666666664   # 1/24

# amp=-0.05, lgb=0 (resubmit of stuck job)
echo "--- amp=-0.05, lgb=0 ---"
sbatch --job-name="osc_lgb0_a-0.05" \
    --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=${A_MG},B_MG=${B_MG},MIN_DR=${MIN_DR},AMP=-0.05 \
    submit_oscillon.sh 0

# amp=-0.1, lgb=5,6,7,8
echo "--- amp=-0.1, lgb=5-8 ---"
for LGB in 5 6 7 8; do
    sbatch --job-name="osc_lgb${LGB}_a-0.1" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=${A_MG},B_MG=${B_MG},MIN_DR=${MIN_DR},AMP=-0.1 \
        submit_oscillon.sh "${LGB}"
done

# amp=-0.5, lgb=0-8
echo "--- amp=-0.5, lgb=0-8 ---"
for LGB in 0 1 2 3 4 5 6 7 8; do
    sbatch --job-name="osc_lgb${LGB}_a-0.5" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=${A_MG},B_MG=${B_MG},MIN_DR=${MIN_DR},AMP=-0.5 \
        submit_oscillon.sh "${LGB}"
done

# amp=-1, lgb=0-8
echo "--- amp=-1, lgb=0-8 ---"
for LGB in 0 1 2 3 4 5 6 7 8; do
    sbatch --job-name="osc_lgb${LGB}_a-1" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=${A_MG},B_MG=${B_MG},MIN_DR=${MIN_DR},AMP=-1 \
        submit_oscillon.sh "${LGB}"
done

echo "Done. Check queue with:  squeue -u \$USER --cluster=genius"
