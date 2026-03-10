#!/bin/bash -l
#
# Submit oscillon runs: mu=0.08, beta=0, T=800, amp=-0.05, lambda_GB = 0..15
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p slurm_output

MU=0.08
T=800
COUPLING=quadratic_0
AMP=-5e-2

LAMBDAS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

for LGB in "${LAMBDAS[@]}"; do
    echo "Submitting lambda_GB = ${LGB}  (mu=${MU}, beta=0, T=${T}, amp=${AMP}) ..."
    sbatch --job-name="osc_lgb${LGB}_b0_a05" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},AMP=${AMP} \
        submit_oscillon.sh "${LGB}"
done

echo "All jobs submitted.  Check queue with:  squeue -u \$USER --cluster=genius"
