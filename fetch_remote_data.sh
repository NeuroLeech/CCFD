#!/usr/bin/env bash
# Fetch what a fresh checkout needs for the unfiltered-target fits (a cloud session, say).
#
# 1. The DATA BUNDLE, ~1.9 GB, fetched once: the unfiltered RBC group FC and its five band FCs
#    (results/fc/nobp/), the 100-channel input profiles (results/xspec_asc_first_area_100.npz),
#    and the small caches that are slow or impossible to rebuild without neuromaps and
#    wb_command (cortical maps, medoids, the 6 mm smoothing matrix, the mesh). Checksummed.
# 2. The SHARED FOLDER, small, fetched every time: Dropbox/CCFD-remote, laid out like the repo
#    (results/xspec_<tag>.npz model solutions, and whatever else is put there). Its link is
#    fixed, so anything added to the folder arrives here with no new link.
#
#   bash fetch_remote_data.sh            # both
#   bash fetch_remote_data.sh --folder   # the shared folder only
set -euo pipefail
cd "$(dirname "$0")"

DATA_URL="https://www.dropbox.com/scl/fi/onew59w1e66k4x3ueg7lj/ccfd_remote_data.zip?rlkey=xxtn576aujkn1u7a6t570p6gl&st=dxbdmcvb&dl=1"
DATA_SHA="ded6701125ba99c5585479cd5eaa988c57a37c3427750734137751ecfd18fae2"
FOLDER_URL="https://www.dropbox.com/scl/fo/rnv1hzwj7e2gqfrsbz8il/AO3Sz2x17XSyHLGntnvL0Qw?rlkey=35t0kw108w5f683o0a6n7koy5&dl=1"

# Dropbox's folder archives carry a bare "/" entry that makes unzip exit non-zero, so both
# archives are extracted with zipfile, skipping anything absolute or escaping the repo.
extract() {
  python3 - "$1" <<'PY'
import sys, zipfile, os
root = os.path.abspath(".")
with zipfile.ZipFile(sys.argv[1]) as z:
    n = 0
    for m in z.infolist():
        name = m.filename.lstrip("/")
        dest = os.path.abspath(os.path.join(root, name))
        if not name or not dest.startswith(root + os.sep):
            continue
        if m.is_dir():
            os.makedirs(dest, exist_ok=True); continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(m) as src, open(dest, "wb") as out:
            out.write(src.read())
        n += 1
print(f"  {n} files from {sys.argv[1]}")
PY
}

if [ "${1:-}" != "--folder" ]; then
  ZIP=ccfd_remote_data.zip
  [ -f "$ZIP" ] || curl -L --fail -o "$ZIP" "$DATA_URL"
  if command -v shasum >/dev/null; then echo "$DATA_SHA  $ZIP" | shasum -a 256 -c
  else echo "$DATA_SHA  $ZIP" | sha256sum -c; fi
  extract "$ZIP"
fi

TMP=$(mktemp -t ccfd_folder.XXXXXX).zip
curl -L --fail -o "$TMP" "$FOLDER_URL"
extract "$TMP"
rm -f "$TMP"
