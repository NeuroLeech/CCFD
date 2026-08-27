"""Write a double-centred copy of an FC matrix, with its vertices and metadata alongside.

FCTarget can centre on the fly (centre='double'), so this is only needed when a centred
matrix is wanted as a file - for plotting, for sharing, or to avoid re-centring a large
matrix on every run.

  python fc_centre.py results/fc/group-NKI100_hemi-L_space-fsaverage5_mask-glasser_spearmanfc.npy
"""
import os, json, shutil, argparse
import numpy as np

from fc_score import double_centre, vertices_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fc_path")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    a = ap.parse_args()

    FC = np.load(a.fc_path)
    n = FC.shape[0]
    iu = np.triu_indices(min(n, 3000), 1)
    before = FC[:3000, :3000][iu]
    FC = double_centre(FC, inplace=True).astype(a.dtype, copy=False)
    after = FC[:3000, :3000][iu]

    out = a.fc_path.replace("fc.npy", "dcfc.npy")
    if out == a.fc_path:
        raise ValueError(f"unexpected name, refusing to overwrite: {a.fc_path}")
    np.save(out, FC)
    vin, vout = vertices_path(a.fc_path), vertices_path(out)
    if os.path.abspath(vin) != os.path.abspath(vout):     # both names usually resolve to
        shutil.copy(vin, vout)                            # the same companion file

    meta_in = a.fc_path.replace("_spearmanfc.npy", "_meta.json").replace(
        "_pearsonfc.npy", "_meta.json")
    if os.path.exists(meta_in):
        m = json.load(open(meta_in))
        m["centring"] = ("double centred: row and column means removed, grand mean added "
                         "back, diagonal excluded from the means and left at 1")
        m["source_fc"] = os.path.basename(a.fc_path)
        m["fc_file"] = os.path.basename(out)
        json.dump(m, open(out.replace("dcfc.npy", "dcfc_meta.json"), "w"), indent=2)

    print(f"  {os.path.basename(a.fc_path)}  ->  {os.path.basename(out)}   {FC.shape} {FC.dtype}")
    print(f"  edge values: mean {before.mean():+.3f} -> {after.mean():+.3f}, "
          f"sd {before.std():.3f} -> {after.std():.3f}, "
          f"negative {100*(before<0).mean():.0f}% -> {100*(after<0).mean():.0f}%")


if __name__ == "__main__":
    main()
