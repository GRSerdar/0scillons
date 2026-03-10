#!/bin/bash -l
#
# Submit oscillon runs: mu=0.08, beta=50, T=800, lambda_GB = 1, 2, 3, 4, 5, 6, 7, 8, 9
# Uses --export so the job actually gets COUPLING=quadratic_50 (sbatch does not pass env by default).
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p slurm_output

MU=0.08
T=800
COUPLING=quadratic_50
# NPTS=1000 (default) => 1000 output points over T=800

LAMBDAS=(1 2 3 4 5 6 7 8 9)

for LGB in "${LAMBDAS[@]}"; do
    echo "Submitting lambda_GB = ${LGB}  (mu=${MU}, beta=50, T=${T}) ..."
    sbatch --job-name="osc_lgb${LGB}_b50" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING} \
        submit_oscillon.sh "${LGB}"
done

echo "All jobs submitted.  Check queue with:  squeue -u \$USER --cluster=genius"
