#!/bin/bash -l
#
# Submit sinh spacing runs with explicit a parameter:
#   3 gauges × 2 resolutions × 2 sinh_a values = 12 runs
#
# lambda=0 (GR), mu=0.08, A=-0.02, a=0.2, b=0.4, T=800
#

cd "$(dirname "$0")"

LGB=0; MU=0.08; AMP=-2e-2; A_MG=0.2; B_MG=0.4; T=800; NPTS=1000

GAUGES=("modified_harmonic" "baumgarte" "bona_masso")
GAUGE_SHORT=("mh" "bg" "bm")
MIN_DRS=("0.125" "0.08333333")
RES_SHORT=("lo" "hi")
SINH_AS=("0.5" "0.05")

COUNT=0
for sa_idx in "${!SINH_AS[@]}"; do
    SA="${SINH_AS[$sa_idx]}"
    for g_idx in "${!GAUGES[@]}"; do
        G="${GAUGES[$g_idx]}"
        GS="${GAUGE_SHORT[$g_idx]}"
        for r_idx in "${!MIN_DRS[@]}"; do
            DR="${MIN_DRS[$r_idx]}"
            RS="${RES_SHORT[$r_idx]}"
            JOB="gt_${GS}_${RS}_sa${SA}"
            echo "Submitting: ${G}, dr=${DR}, sinh_a=${SA}"
            sbatch --job-name="${JOB}" \
                --export=ALL,LGB=${LGB},MU=${MU},AMP=${AMP},A_MG=${A_MG},B_MG=${B_MG},T=${T},NPTS=${NPTS},GAUGE=${G},MIN_DR=${DR},SPACING=sinh,SINH_A=${SA},FORCE=1 \
                submit_gauge_test.sh 0
            COUNT=$((COUNT + 1))
        done
    done
done

echo ""
echo "Submitted ${COUNT} sinh_a comparison jobs."
