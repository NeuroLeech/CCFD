"""Diagnostics behind the A4 note: how much of each realised observable is one global
signal, and how large its correlations are against the target's. Reads the draw-0 frames
each build saves (results/frames_<tag>.npy).

  python docs/framing/logs/a4/global_share.py
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path[:0] = [os.path.join(ROOT, d) for d in ("core", "fit", "targets")]
os.chdir(ROOT)
import numpy as np
from mesh_cache import load_cortex
from fc_score import align_to_cortex, vertices_path

TAGS = ["nobp_sm6_s1_box", "nobp_sm6_s3_box", "nobp_sm6_reach033_box", "nobp_sm6_reach3_box",
        "nobp_sm6_s3_hrf_box", "nobp_sm6_s3_hrf_f600_box"]
FC = "results/fc/nobp/group-RBCNKI100nobp_hemi-L_space-fsaverage5_mask-glasser_spearmanfc.npy"

c = load_cortex("fsaverage5", verbose=False)
fc, cols, _ = align_to_cortex(np.asarray(np.load(FC, mmap_mode="r")), np.load(vertices_path(FC)), c)
n = fc.shape[0]
print(f"target: correlation participation ratio N^2/sum C^2 = {n * n / float((fc.astype(np.float64) ** 2).sum()):.1f}")
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(n, 1500, replace=False))
iu = np.triu_indices(len(sel), 1)
e = np.asarray(fc[np.ix_(sel, sel)], float)[iu]
print(f"target: mean correlation {e.mean():+.3f} (1,500 random vertices)")
for tag in TAGS:
    f = f"results/frames_{tag}.npy"
    if not os.path.exists(f):
        continue
    X = np.load(f).astype(np.float64)[50:]
    X = X - X.mean(0)
    sd = X.std(0)
    X = X[:, sd > 0] / sd[sd > 0]
    ev = np.clip(np.linalg.eigvalsh(X @ X.T), 0, None)
    g = X.mean(1)
    g /= np.linalg.norm(g)
    r2 = (g @ X) ** 2 / (X ** 2).sum(0)
    Xs = np.load(f).astype(np.float64)[50:][:, np.asarray(cols)[sel]]
    Xs = (Xs - Xs.mean(0)) / Xs.std(0)
    r = ((Xs.T @ Xs) / len(Xs))[iu]
    print(f"{tag:28s}: correlation PR {ev.sum() ** 2 / (ev ** 2).sum():5.1f}   global signal "
          f"{r2.mean():5.1%} of variance   mean correlation {r.mean():+.3f}")
