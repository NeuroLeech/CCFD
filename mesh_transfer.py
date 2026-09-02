"""Do the two meshes give the same TRANSFER FUNCTION in the fMRI band?

mesh_check found the field statistics converged (autocorrelation half-distance 7.62 mm on
fsaverage5, 7.51 on fsaverage6) while the individual impulse responses decorrelate with
time - +0.95 at one frame, +0.39 by the 9 s decay. Pointwise divergence in a linear system
is dispersion error plus the small metric difference between the two inflations; it says
nothing directly about the fit, because the solve never sees a response in the time domain.
It sees H(f) at 0.01-0.1 Hz, where each bin integrates hundreds of frames.

So the question that decides whether an fsaverage6 re-run could move the score is whether
H(f) agrees THERE. Complex correlation per frequency, over (piece, vertex), on the shared
vertices, with the same zero-padding the solve uses.

  python mesh_transfer.py
"""
import os, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import bo_step, xspec, subparcels, ladder, timescale, fluid as fl
from best_fit import BEST_X


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pad", type=int, default=4096)
    ap.add_argument("--regions", default="subcortical")
    ap.add_argument("--split", type=int, default=40)
    ap.add_argument("--nframe", type=int, default=224)
    a = ap.parse_args()

    c5 = load_cortex("fsaverage5", verbose=False)
    c6 = load_cortex("fsaverage6", verbose=False)
    clock = timescale.plan(4, decay_s=timescale.BOLD_TAU_S, spread_mm_s=6.0, verbose=False)
    x = BEST_X.copy(); x[3] = np.log10(clock["save"]); x[0] = np.log10(clock["damp"])
    p, save5, _ = bo_step.unpack(x, c5)
    f5, _ = fl.fields(c5, p); f6, _ = fl.fields(c6, p)
    save6 = int(round(save5 * (fl.CFL * c5.d.min() / f5.max())
                      / (fl.CFL * c6.d.min() / f6.max())))
    parcels, total = subparcels.region_set(c5, a.regions, a.split)
    lab5, tags = subparcels.split_parcels(c5, parcels, total, verbose=False)
    P5 = subparcels.taper_profiles(c5, lab5, len(tags))
    _, V5 = ladder._white_graph(c5); _, V6 = ladder._white_graph(c6)
    from scipy.spatial import cKDTree
    P6 = np.ascontiguousarray(P5[:, cKDTree(V5).query(V6)[1]])

    R5 = xspec.impulse_responses(c5, list(range(len(tags))), p, a.nframe * save5, save5,
                                 profiles=P5, verbose=False)
    R6 = xspec.impulse_responses(c6, list(range(len(tags))), p, a.nframe * save6, save6,
                                 profiles=P6, verbose=False)
    o5 = np.asarray(c5.old); o6 = np.asarray(c6.old)
    common = np.intersect1d(o5, o6)
    i5 = np.searchsorted(o5, common); i6 = np.searchsorted(o6, common)

    frame_s = clock["frame_s"]
    nb = a.pad // 2 + 1
    want = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15,
                     0.25, 0.4, 0.6, 0.775])
    bins = np.unique(np.round(want * a.pad * frame_s).astype(int))
    bins = bins[(bins >= 1) & (bins < nb)]
    hz = bins / (a.pad * frame_s)
    print(f"  {len(tags)} pieces, pad {a.pad} ({a.pad*frame_s:.0f} s), "
          f"{len(common)} shared vertices")

    num = np.zeros(len(bins), np.complex128)
    d5 = np.zeros(len(bins)); d6 = np.zeros(len(bins))
    pw5 = np.zeros(len(bins)); pw6 = np.zeros(len(bins))
    for k in range(len(tags)):
        H5 = np.fft.rfft(np.asarray(R5[k][:, i5], np.float64), n=a.pad, axis=0)[bins]
        H6 = np.fft.rfft(np.asarray(R6[k][:, i6], np.float64), n=a.pad, axis=0)[bins]
        num += np.einsum("fv,fv->f", H5.conj(), H6)
        d5 += np.einsum("fv,fv->f", H5.conj(), H5).real
        d6 += np.einsum("fv,fv->f", H6.conj(), H6).real
        pw5 += np.abs(H5).mean(1); pw6 += np.abs(H6).mean(1)
        del H5, H6
    r = np.abs(num) / np.sqrt(d5 * d6)
    ph = np.angle(num)

    print(f"\n  transfer-function agreement between meshes")
    print(f"    {'Hz':>9s}{'|corr|':>9s}{'phase':>9s}{'fs6/fs5 gain':>14s}")
    for i, f in enumerate(hz):
        print(f"    {f:9.4f}{r[i]:9.4f}{np.degrees(ph[i]):8.1f}d"
              f"{pw6[i]/max(pw5[i],1e-30):14.3f}")
    band = (hz >= 0.01) & (hz <= 0.1)
    print(f"\n  resting-state band 0.01-0.1 Hz: mean |corr| {r[band].mean():.4f}")
    print(f"  above the band: mean |corr| {r[~band & (hz>0.1)].mean():.4f}")
    np.savez(os.path.join(RESULTS, "mesh_transfer.npz"), hz=hz, r=r, phase=ph,
             gain5=pw5, gain6=pw6)
    print(f"  wrote results/mesh_transfer.npz")


if __name__ == "__main__":
    main()
