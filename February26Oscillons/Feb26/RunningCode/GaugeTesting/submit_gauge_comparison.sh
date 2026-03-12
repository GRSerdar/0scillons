#!/bin/bash -l
#
# Submit gauge comparison: modified_harmonic vs baumgarte
# at two resolutions (low: dr=1/8, high: dr=1/12)
# lambda=0 (GR), mu=0.08, A=-0.02, a=0.2, b=0.4, T=800
#

cd "$(dirname "$0")"

LGB=0; MU=0.08; AMP=-2e-2; A_MG=0.2; B_MG=0.4; T=800; NPTS=1000

# ── modified_harmonic, low resolution (dr = 1/8 = 0.125) ──
echo "Submitting: modified_harmonic, dr=0.125"
sbatch --job-name="gt_mh_lo" \
    --export=ALL,LGB=${LGB},MU=${MU},AMP=${AMP},A_MG=${A_MG},B_MG=${B_MG},T=${T},NPTS=${NPTS},GAUGE=modified_harmonic,MIN_DR=0.125 \
    submit_gauge_test.sh 0

# ── modified_harmonic, high resolution (dr = 1/12 ≈ 0.08333) ──
echo "Submitting: modified_harmonic, dr=0.08333"
sbatch --job-name="gt_mh_hi" \
    --export=ALL,LGB=${LGB},MU=${MU},AMP=${AMP},A_MG=${A_MG},B_MG=${B_MG},T=${T},NPTS=${NPTS},GAUGE=modified_harmonic,MIN_DR=0.08333333 \
    submit_gauge_test.sh 0

# ── baumgarte, low resolution (dr = 1/8 = 0.125) ──
echo "Submitting: baumgarte, dr=0.125"
sbatch --job-name="gt_bg_lo" \
    --export=ALL,LGB=${LGB},MU=${MU},AMP=${AMP},A_MG=${A_MG},B_MG=${B_MG},T=${T},NPTS=${NPTS},GAUGE=baumgarte,MIN_DR=0.125 \
    submit_gauge_test.sh 0

# ── baumgarte, high resolution (dr = 1/12 ≈ 0.08333) ──
echo "Submitting: baumgarte, dr=0.08333"
sbatch --job-name="gt_bg_hi" \
    --export=ALL,LGB=${LGB},MU=${MU},AMP=${AMP},A_MG=${A_MG},B_MG=${B_MG},T=${T},NPTS=${NPTS},GAUGE=baumgarte,MIN_DR=0.08333333 \
    submit_gauge_test.sh 0

echo ""
echo "Submitted 4 gauge-testing jobs."
