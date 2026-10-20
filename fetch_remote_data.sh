#!/usr/bin/env bash
# Fetch the data a fresh checkout needs for the unfiltered-target fits (a cloud session, say):
# the unfiltered RBC group FC and its five band FCs (results/fc/nobp/), the 100-channel input
# profiles (results/xspec_asc_first_area_100.npz), and the small caches that are slow or
# impossible to rebuild without neuromaps and wb_command (cortical maps, medoids, the 6 mm
# smoothing matrix, the mesh). ~1.9 GB, hosted on Dropbox; built 2026-10-20 from 0a4ce15.
#
#   bash fetch_remote_data.sh
set -euo pipefail
cd "$(dirname "$0")"
URL="https://www.dropbox.com/scl/fi/onew59w1e66k4x3ueg7lj/ccfd_remote_data.zip?rlkey=xxtn576aujkn1u7a6t570p6gl&st=dxbdmcvb&dl=1"
SHA="ded6701125ba99c5585479cd5eaa988c57a37c3427750734137751ecfd18fae2"
ZIP=ccfd_remote_data.zip
if [ ! -f "$ZIP" ]; then
  curl -L --fail -o "$ZIP" "$URL"
fi
if command -v shasum >/dev/null; then
  echo "$SHA  $ZIP" | shasum -a 256 -c
else
  echo "$SHA  $ZIP" | sha256sum -c
fi
unzip -o -q "$ZIP"
echo "unpacked: results/fc/nobp/, results/xspec_asc_first_area_100.npz, data/cache/ (maps, medoids, smoothing, mesh)"
