#!/usr/bin/env bash
# Rotation in the linear fit: 3x speed, 8.33 s decay, inertial periods 10, 30, 100 s.
# Each: one fit (unfiltered target, 150 evaluations), its realised score, an unfiltered video.
set -u
cd /home/user/CCFD
S=/tmp/claude-0/-home-user-CCFD/afe0c887-85f1-5c5c-868e-0abf11c14e63/scratchpad
Q=$S/queue; ST=$Q/status
step() {
  local name=$1; shift
  if grep -q "END   $name rc=0" "$ST"; then return; fi
  echo "$(date '+%F %T') START $name" >> "$ST"
  "$@" > "$Q/$name.log" 2>&1
  echo "$(date '+%F %T') END   $name rc=$?" >> "$ST"
}
COMMON="--oversample 4 --impulse-decays 7 --pad 4096 --nfreq 192 --bold-smooth \
  --regions hybrid --split 50 --nvert 2000 --val-vert 0 --solver factor --rank 20 \
  --seconds 577 --draws 4 --workers 3 --hybrid-file results/xspec_asc_first_area_100.npz \
  --fc-path results/fc/nobp/group-RBCNKI100nobp_hemi-L_space-fsaverage5_mask-glasser_spearmanfc.npy \
  --smooth-fwhm 6 --decimate mean --maxfun 150 --maps-scale 0"
for T in 30 10 100; do
  tag=nobp_sm6_s3_rot${T}_box
  step build_$tag python3 fit/best_fit.py $COMMON --spread-mm-s 4.5 --decay-s 8.3333 \
    --inertial-s $T --tag $tag
  step frames_$tag env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 fit/rescore.py \
    --tag $tag --seconds 577 --draws 1 --workers 1 --save-frames
  step video_$tag python3 viz/render_bandpass.py --tag ${tag}_577s \
    --raw-npy results/frames_${tag}_577s_raw.npy --band none --smooth-fwhm 6 \
    --start 200 --n 600 --save 12 --out results/videos/${tag}_unfiltered.mp4
done
echo "$(date '+%F %T') ROTATION DONE" >> "$ST"
