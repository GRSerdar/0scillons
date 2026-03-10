#!/bin/bash -l
#
# Submit oscillon runs: mu=0.08, beta=10, T=800, lambda_GB = 6, 7, 8, 9, 10
# Uses --export so the job actually gets COUPLING=quadratic_10.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p slurm_output

MU=0.08
T=800
COUPLING=quadratic_10

LAMBDAS=(6 7 8 9 10)

for LGB in "${LAMBDAS[@]}"; do
    echo "Submitting lambda_GB = ${LGB}  (mu=${MU}, beta=10, T=${T}) ..."
    sbatch --job-name="osc_lgb${LGB}_b10" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING} \
        submit_oscillon.sh "${LGB}"
done

echo "All jobs submitted.  Check queue with:  squeue -u \$USER --cluster=genius"
