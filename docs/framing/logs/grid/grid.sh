#!/usr/bin/env bash
# Speed x decay grid: each cell one fit (unfiltered target, 150 evaluations, uniform
# medium), its realised score, and an unfiltered video. Runs once the map-weighted rebuild
# has finished; then the rest of queue5 (minus the 664 mm refit) resumes.
set -u
cd /home/user/CCFD
S=/tmp/claude-0/-home-user-CCFD/afe0c887-85f1-5c5c-868e-0abf11c14e63/scratchpad
Q=$S/queue; ST=$Q/status; V=$S/videos
step() {
  local name=$1; shift
  if grep -q "END   $name rc=0" "$ST"; then return; fi
  echo "$(date '+%F %T') START $name" >> "$ST"
  "$@" > "$Q/$name.log" 2>&1
  echo "$(date '+%F %T') END   $name rc=$?" >> "$ST"
}
# wait for the map-weighted rebuild, then stop queue5 at that step boundary
until grep -q "END   build_mapsmm_s3" "$ST"; do sleep 10; done
pkill -f "bash $S/queue5.sh"; pkill -f "analysis/interplay.py bold --tag nobp_sm6_mapsmm_s3_box"
sleep 5
COMMON="--oversample 4 --impulse-decays 7 --pad 4096 --nfreq 192 --bold-smooth \
  --regions hybrid --split 50 --nvert 2000 --val-vert 0 --solver factor --rank 20 \
  --seconds 577 --draws 4 --workers 3 --hybrid-file results/xspec_asc_first_area_100.npz \
  --fc-path results/fc/nobp/group-RBCNKI100nobp_hemi-L_space-fsaverage5_mask-glasser_spearmanfc.npy \
  --smooth-fwhm 6 --decimate mean"
cell() {   # speed label, decay s
  local v=$1 d=$2 tag=nobp_sm6_v$1_d$2_box
  step build_$tag python3 fit/best_fit.py $COMMON --maxfun 150 --maps-scale 0 \
    --spread-mm-s $v --decay-s $d --tag $tag
  local save=$(python3 -c "import numpy as np;print(int(np.load('results/xspec_$tag.npz')['save']))")
  step frames_$tag env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 fit/rescore.py \
    --tag $tag --seconds 577 --draws 1 --workers 1 --save-frames
  step video_$tag python3 viz/render_bandpass.py --tag ${tag}_577s \
    --raw-npy results/frames_${tag}_577s_raw.npy --band none --smooth-fwhm 6 \
    --start 200 --n 600 --save $save --out results/videos/${tag}_unfiltered.mp4
}
cell 4.5 2.5
cell 4.5 5
cell 1.5 15
cell 1.5 2.5
cell 1.5 5
cell 7.5 2.5
cell 4.5 15
cell 7.5 8.3333
cell 4.5 25
cell 7.5 15
echo "$(date '+%F %T') GRID DONE" >> "$ST"
exec bash $S/queue5b.sh
