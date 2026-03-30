#!/bin/bash -l
#
# Submit oscillon runs: mu=0.08, linear coupling, a=b=0, T=800
# lambda_GB = 1, 2, 3, 4, 5, 6, 7, 8, 9
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p slurm_output

MU=0.08
T=800
COUPLING=linear
A_MG=0
B_MG=0

LAMBDAS=(1 2 3 4 5 6 7 8 9)

for LGB in "${LAMBDAS[@]}"; do
    echo "Submitting lambda_GB = ${LGB}  (mu=${MU}, linear, a=b=0, T=${T}) ..."
    sbatch --job-name="osc_lgb${LGB}_lin_ab0" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=${A_MG},B_MG=${B_MG} \
        submit_oscillon.sh "${LGB}"
done

echo "All jobs submitted.  Check queue with:  squeue -u \$USER --cluster=genius"
