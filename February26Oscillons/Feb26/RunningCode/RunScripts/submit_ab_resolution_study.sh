#!/bin/bash -l
#
# Resolution study with nonzero gauge parameters
# a=0.2,b=0.4 and a=0.1,b=0.2 at dr=1/24 and dr=1/32
# lambda_GB = 0, 1, 2 — amp=-0.02 — quadratic beta=0 — T=800
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p slurm_output

MU=0.08
T=800
COUPLING=quadratic_0
AMP=-0.02

LAMBDAS=(0 1 2)
DR_24=0.041666666666666664   # 1/24
DR_32=0.03125                # 1/32

# a=0.2, b=0.4, dr=1/24
for LGB in "${LAMBDAS[@]}"; do
    echo "Submitting lgb=${LGB} a=0.2 b=0.4 dr=1/24 ..."
    sbatch --job-name="osc_lgb${LGB}_a02b04_24" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=0.2,B_MG=0.4,MIN_DR=${DR_24},AMP=${AMP} \
        submit_oscillon.sh "${LGB}"
done

# a=0.2, b=0.4, dr=1/32
for LGB in "${LAMBDAS[@]}"; do
    echo "Submitting lgb=${LGB} a=0.2 b=0.4 dr=1/32 ..."
    sbatch --job-name="osc_lgb${LGB}_a02b04_32" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=0.2,B_MG=0.4,MIN_DR=${DR_32},AMP=${AMP} \
        submit_oscillon.sh "${LGB}"
done

# a=0.1, b=0.2, dr=1/24
for LGB in "${LAMBDAS[@]}"; do
    echo "Submitting lgb=${LGB} a=0.1 b=0.2 dr=1/24 ..."
    sbatch --job-name="osc_lgb${LGB}_a01b02_24" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=0.1,B_MG=0.2,MIN_DR=${DR_24},AMP=${AMP} \
        submit_oscillon.sh "${LGB}"
done

# a=0.1, b=0.2, dr=1/32
for LGB in "${LAMBDAS[@]}"; do
    echo "Submitting lgb=${LGB} a=0.1 b=0.2 dr=1/32 ..."
    sbatch --job-name="osc_lgb${LGB}_a01b02_32" \
        --export=ALL,MU=${MU},T=${T},COUPLING=${COUPLING},A_MG=0.1,B_MG=0.2,MIN_DR=${DR_32},AMP=${AMP} \
        submit_oscillon.sh "${LGB}"
done

echo "All jobs submitted.  Check queue with:  squeue -u \$USER --cluster=genius"
