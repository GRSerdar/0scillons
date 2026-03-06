#!/bin/bash -l
#
# Submit oscillon simulations for several lambda_GB values.
# Edit the LAMBDAS array to change which runs are submitted.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LAMBDAS=(0.00 0.01 0.03 0.05)

for LGB in "${LAMBDAS[@]}"; do
    echo "Submitting lambda_GB = ${LGB} ..."
    sbatch submit_oscillon.sh "${LGB}"
done

echo "All jobs submitted.  Check queue with:  squeue -u \$USER"
