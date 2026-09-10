"""Read the solved S(f) back as ascending systems: per-nucleus power, coordination,
mode content, and how far each nucleus's power can move at matched fit.

  python analysis/ascending_readout.py --tag ascend_085
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import fc_score, xspec
import onoff_fields


def nuclei(tags):
    return sorted({str(t).split(":")[0] for t in tags})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="ascend_085")
    ap.add_argument("--eps", default="0.002,0.01")
    ap.add_argument("--band", default="0.01,0.08")
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=False)
    z = np.load(os.path.join(RESULTS, f"xspec_{a.tag}.npz"), allow_pickle=True)
    S = np.asarray(z["S"])
    tags = [str(s) for s in z["tags"]]
    nms = nuclei(tags)
    band = tuple(float(v) for v in a.band.split(","))

    # --- per-nucleus input power, mode content, and lag-0 coordination ---------------
    pw = xspec.piece_power(S, z["idx"], float(z["frame_s"]), band=band,
                           pad=int(z["pad"]))
    print(f"\n  input power in {a.band} Hz, by nucleus (and mode split):")
    for nm in nms:
        k = [i for i, tg in enumerate(tags) if tg.split(":")[0] == nm]
        share = pw[k] / max(pw[k].sum(), 1e-30)
        print(f"    {nm:<12s} {pw[k].sum():9.4g}   modes "
              + " ".join(f"{tags[i].split(':')[1]}={share[j]:.2f}"
                         for j, i in enumerate(k)))

    C0 = onoff_fields.to_corr(onoff_fields.input_cov(z, 0.0, band))
    print(f"\n  lag-0 input correlation between nuclei:")
    for i, a_nm in enumerate(nms):
        ka = [k for k, tg in enumerate(tags) if tg.split(":")[0] == a_nm]
        row = []
        for j, b_nm in enumerate(nms):
            if j <= i:
                continue
            kb = [k for k, tg in enumerate(tags) if tg.split(":")[0] == b_nm]
            row.append(f"{b_nm}={C0[np.ix_(ka, kb)].max():+.3f}")
        if row:
            print(f"    {a_nm:<12s} " + "  ".join(row))

    # --- family brackets: how far can each nucleus's power move at matched fit --------
    from zones import rebuild
    R = rebuild(a.tag, c, t)
    H, w = R["H"], R["w"]
    raw = np.asarray(t.target_fc(), np.float64)
    raw = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
    Ct = xspec.normal_scores(raw)
    print(f"\n  family brackets at matched fit (share of total input power):")
    for nm in nms:
        k = [i for i, tg in enumerate(tags) if tg.split(":")[0] == nm]
        share0 = float(pw[k].sum() / pw.sum())
        lo, hi = share0, share0
        for sign in (-1, 1):
            Rg = xspec.share_reg(w, k, H.shape[2])
            S2, rep = xspec.family_member(H, w, Ct, S, Rg, eps=0.01, iters=200,
                                          sign=sign, verbose=False)
            pw2 = xspec.piece_power(S2, z["idx"], float(z["frame_s"]), band=band,
                                    pad=int(z["pad"]))
            v = float(pw2[k].sum() / max(pw2.sum(), 1e-30))
            lo, hi = min(lo, v), max(hi, v)
        print(f"    {nm:<12s} share {share0:.3f} in [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
