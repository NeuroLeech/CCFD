"""Adaptive damping, unoptimised: does it put graded slow structure into a fast medium?

A linear medium can only structure the 0.01-0.08 Hz observable with dynamics on that
timescale, which a realistically fast medium does not have (10x speed with any decay is the
1x medium seen through a 10x slower band). torch_swe.run_adaptive adds one slow state per
vertex - a running average of local wave energy, time constant tau_a - that scales the
damping by 1 + k a / a_ref. This runs it with NO fitting, on two drives that keep the input
territory's spatial footprint:

  bestfit   the best linear fit's drive (nv2000_mf400: 100 regions, 2,000 solve vertices)
  random    the factor solver's random starting cross-spectrum over the same 100 regions -
            same territory and footprint, unfitted timing (not per-vertex noise, which has
            no spatial structure at all)

and reads two observables from the same run, both frame-averaged then BOLD-smoothed and
bandpassed: the local ENERGY h^2 (BOLD read from activity) and the signed height h. Each is
scored against the target FC (Spearman, the headline scorer) and its FC's variogram fitted
(effective range, and how far correlation falls), raw and with the global component removed.

FIRST RUN, 2026-10-06 (577 s, burn 64 s; range and far correlation from the variogram fit):

                                     ENERGY                       HEIGHT
    medium    drive    k  tau    score  range  r_far          score  max gain
    1x        bestfit  0   -     0.059   40    +0.02          0.689    1.0
              bestfit  4  30     0.078   46    +0.03          0.663    5.6
              random   0   -     0.053   30    +0.32          0.255    1.0
              random   4  30     0.063   31    +0.23          0.270    5.2
    10x/0.87  bestfit  0   -     0.244  964    -0.76          0.295    1.0
              bestfit  1  10     0.251  445    -0.22          0.292    2.6
              bestfit  4  10     0.244  253    -0.02          0.282    5.6
              random   0   -     0.100   61    +0.19          0.259    1.0
              random   4  10     0.089   49    +0.02          0.240    5.3
    target FC                          52    ~0

In the fast medium adaptation LOCALISES the energy field - 964 mm with long-range
anticorrelation to 253 mm with none (bestfit), 61 to 49 mm (random, the target's scale) - while
the scores, unfitted, stay put. tau_a 10 vs 30 s barely matters. At 1x it changes little. The
energy observable is far more informative at 10x than at 1x for the same drive (0.244 vs 0.059).

  python fit/adaptive_probe.py --media nv2000_mf400 f10m_d0.87_nv2000_mf150
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np

from paths import RESULTS


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--media', nargs='+', default=['nv2000_mf400', 'f10m_d0.87_nv2000_mf150'])
    ap.add_argument('--drive-tag', default='nv2000_mf400')
    ap.add_argument('--k', type=float, nargs='+', default=[1.0, 4.0])
    ap.add_argument('--tau', type=float, nargs='+', default=[10.0, 30.0])
    ap.add_argument('--seconds', type=float, default=577.0)
    ap.add_argument('--burn-s', type=float, default=64.0)
    ap.add_argument('--device', default='mps')
    a = ap.parse_args()
    import torch, xspec, units, bandpass, timescale, fluid as fl
    from interp_gap import medium_context
    from torch_swe import TorchSWE, run_adaptive
    from fc_score import _rank_z
    from variogram import profile, fit_range, deglobal

    zd = np.load(os.path.join(RESULTS, f'xspec_{a.drive_tag}.npz'), allow_pickle=True)
    idx, ref = np.asarray(zd['idx']), int(zd['ref_frames'])
    kern, c, t, g0 = medium_context(zd)
    nfr = timescale.frames_for(a.seconds, g0['frame_s'])
    import re as _re
    m = _re.search(r'--rank\s+(\d+)', str(zd['argv'])) if 'argv' in zd.files else None
    nf, K, r = len(idx), xspec.solution_K(zd), int(m.group(1)) if m else 20
    half = nf * K * r
    L0 = np.random.default_rng(0).standard_normal(2 * half) * (1.0 / np.sqrt(K * r))
    L0 = list(L0[:half].reshape(nf, K, r) + 1j * L0[half:].reshape(nf, K, r))
    drives = {'bestfit': xspec.realise(None, idx, nfr, ref_frames=ref, seed=1000,
                                       factors=xspec.load_factors(zd)),
              'random': xspec.realise(None, idx, nfr, ref_frames=ref, seed=1000, factors=L0)}
    cols = np.asarray(t.cols)
    vsub = cols[g0['sub']]
    D = units.vertex_geodesic(c, vsub)[:, vsub]
    burn = int(a.burn_s / g0['frame_s'])

    def score(X, fs, lo, hi):
        Y = bandpass.apply(units.smooth_frames(X, kern), fs, lo, hi)[burn:]
        Z, flat = _rank_z(np.ascontiguousarray(np.asarray(Y[:, cols], np.float64).T), rank=True)
        sim = float(t._prep(t.model_edges(Z=Z, flat=flat)[0]) @ t.y)
        R = np.corrcoef(np.asarray(Y[:, vsub], np.float64).T)
        R1, _, s1 = fit_range(*profile(R, D))
        R2, _, s2 = fit_range(*profile(deglobal(R), D))
        return sim, R1, 1 - s1, R2

    print(f"  {'medium':<26s} {'drive':<8s} {'k':>3s} {'tau':>4s}  {'energy: score':>13s} "
          f"{'range':>6s} {'r_far':>6s} {'range-g':>8s}   {'height: score':>13s} {'max gain':>9s}  [s]",
          flush=True)
    for mt in a.media:
        z = np.load(os.path.join(RESULTS, f'xspec_{mt}.npz'), allow_pickle=True)
        _, _, _, g = medium_context(z)
        save, fs = g['save'], g['frame_s']
        s, dt, gg, Hf = fl.build(c, g['p'], sponge=True)
        sw = TorchSWE(s, dtype=torch.float32, device=a.device)
        sps = fs / save
        for dn, Af in drives.items():
            Aser = np.repeat(Af, save, axis=0)[:nfr * save] / save
            d = xspec.ProfileDrive(c, g['P'], Aser, 2e-4)
            a_ref = None
            for k, tau in [(0.0, 1.0)] + [(k, tau) for k in a.k for tau in a.tau]:
                t0 = time.time()
                out, _ = run_adaptive(sw, d.Aser, d.P, dt, gg, Hf, save, k=k,
                                      tau_steps=tau / sps, a_ref=(a_ref or 1.0))
                if a_ref is None:
                    a_ref = float(out['energy'][burn:].mean())
                se, Re, rfar, Rg = score(out['energy'], fs, g['lo'], g['hi'])
                sh, *_ = score(out['h'], fs, g['lo'], g['hi'])
                mg = float(np.percentile(1 + k * out['a'][burn:].mean(0) / a_ref, 99)) if k else 1.0
                print(f"  {mt:<26s} {dn:<8s} {k:>3g} {tau if k else 0:>4g}  {se:>+13.4f} "
                      f"{Re:>6.0f} {rfar:>+6.2f} {Rg:>8.0f}   {sh:>+13.4f} {mg:>9.2f}  "
                      f"[{time.time()-t0:.0f}]", flush=True)


if __name__ == '__main__':
    main()
