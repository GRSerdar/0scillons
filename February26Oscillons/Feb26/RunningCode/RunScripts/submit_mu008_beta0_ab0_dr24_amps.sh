#!/bin/bash -l
#
# Submit oscillon runs: mu=0.08, quadratic beta=0, a=b=0, min_dr=1/24, T=800
# lambda_GB = 0,1,2,3,4,5,6,7,8
# amplitudes = -0.05, -0.1, -0.5, -1
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

LAMBDAS=(0 1 2 3 4 5 6 7 8)
AMPS=(-0.05 -0.1 -0.5 -1)

for AMP in "${AMPS[@]}"; do
    for LGB in "${LAMBDAS[@]}"; do
        echo "Submitting lambda_GB=${LGB}, amp=${AMP}  (mu=${MU}, beta=0, a=b=0, dr=1/24, T=${T}) ..."
        sbatch --job-name="osc_lgb${LGB}_a${AMP}" \
            --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=${A_MG},B_MG=${B_MG},MIN_DR=${MIN_DR},AMP=${AMP} \
            submit_oscillon.sh "${LGB}"
    done
done

echo "All jobs submitted.  Check queue with:  squeue -u \$USER --cluster=genius"
