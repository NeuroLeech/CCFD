"""Does the out-of-band part of the solved input do anything?

The objective has no gradient outside the passband: an LTI medium moves no power between
frequencies and H carries the kernel and the passband, so H ~ 0 above 0.08 Hz and whatever S
holds there is untouched initialisation rather than a solved quantity. If that is right, then
drawing the drive from the in-band bins alone and zeroing the rest should give the same FC -
approximately, since the passband is not a brick wall and some energy just outside it leaks
through.

That matters beyond bookkeeping. The project's interest is in a fast medium forced to produce
slow observed structure. If the solved input only matters in-band then the drive is a SLOW
drive, and the model is not reconciling fast input with slow patterns - it is being driven
slowly and the medium's speed is doing less than the framing assumes.

Both conditions use the SAME seeds, the same medium and the same observable, so the only
difference is which bins of S the drive is drawn from. FC is scale invariant and realise
renormalises, so zeroing most of the power is not a confound: the in-band part is simply
amplified to the same rms, which for a linear medium is a constant on the field.

WHAT IT FOUND, on grclip400_nolag over 577 s with matched seeds:

    power zeroed      78.2% of sum_f w_f tr(S_f)
    full S            sim +0.6128 +- 0.0012
    in-band S only    sim +0.5986 +- 0.0047
    difference        -0.0142

So 78% of the drive's power can be deleted for 0.014 of Spearman. The out-of-band part of the
solved S is residue rather than a working part of the model, which is what fit/band_structure.py
shows from the other side: 55 of 100 effective input modes left untouched above the band.

For the linear solve, then, the drive that matters is a SLOW drive. The medium's speed governs how
that slow input spreads over the sheet, but the input itself lives inside the measurement band, so
the fast-medium/slow-pattern reconciliation is not happening in these fits - there is nothing fast
in the part of the input that affects the score.

The complement is untested: the envelope observable exists to give out-of-band input a role, and
band_structure.py measures its above-band S at 4.5 effective modes against the linear's 55, so the
same deletion run on an envelope solve would say whether that structure is load-bearing.

  python fit/band_only_drive.py grclip400_nolag --seconds 577 --draws 2
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np

from mesh_cache import load_cortex
from paths import RESULTS
import xspec, bo_step, timescale, units, fc_score
import bandpass as bp


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tag')
    ap.add_argument('--seconds', type=float, default=577.0)
    ap.add_argument('--draws', type=int, default=2)
    a = ap.parse_args()

    z = np.load(os.path.join(RESULTS, f'xspec_{a.tag}.npz'), allow_pickle=True)
    S, idx, w = z['S'], np.asarray(z['idx'], np.int64), np.asarray(z['H_w'], float)
    pad, save, fs = int(z['pad']), int(z['save']), float(z['frame_s'])
    lo, hi = (float(v) for v in z['band'])
    P = np.asarray(z['profiles'], np.float32)
    c = load_cortex('fsaverage5', verbose=False)
    t = fc_score.default_target(c, verbose=False)
    p, _, _ = bo_step.unpack(np.asarray(z['x'], float), c)
    p['map_clip'] = str(z['map_clip']) if 'map_clip' in z.files else 'none'
    kern = units.smoothing_kernel(timescale.bold_fwhm_frames(fs, verbose=False),
                                  verbose=False)

    f_hz = idx / (pad * fs)
    inb = (f_hz >= lo) & (f_hz <= hi)
    tr = np.array([np.trace(S[f]).real for f in range(len(idx))]) * w
    Sb = S.copy()
    Sb[~inb] = 0.0
    print(f'  {a.tag}: {len(idx)} solved bins, {int(inb.sum())} in {lo}-{hi} Hz')
    print(f'  out-of-band power: {100*(1 - tr[inb].sum()/tr.sum()):.1f}% of sum_f w_f tr(S_f)',
          flush=True)

    # 1/f FILL. Out-of-band bins get tr(S_f) proportional to 1/f with an IDENTITY spatial
    # structure - uncorrelated across regions, the neutral choice, since the data cannot
    # determine that structure either. Scaled so the out-of-band power matches what the solve
    # had, so this differs from `full S` only in the SHAPE of the out-of-band input and not in
    # how much of it there is.
    Sf = S.copy()
    K = S.shape[1]
    oob = ~inb
    dens = 1.0 / np.maximum(f_hz[oob], 1e-12)
    raw = (w[oob] * dens * K)                      # power each bin would carry at tr = dens*K
    Sf[oob] = (dens * (tr[oob].sum() / raw.sum()))[:, None, None] * np.eye(K)[None]
    trf = np.array([np.trace(Sf[i]).real for i in range(len(idx))]) * w
    print(f'  1/f fill: out-of-band power {100*(1 - trf[inb].sum()/trf.sum()):.1f}%, '
          f'tr(S) ratio lowest/highest out-of-band bin '
          f'{np.trace(Sf[np.flatnonzero(oob)[0]]).real/np.trace(Sf[np.flatnonzero(oob)[-1]]).real:.1f}x',
          flush=True)

    import fluid as fl
    from xspec import ProfileDrive
    nframes = timescale.frames_for(a.seconds, fs)
    cols = np.asarray(t.cols)
    out = {}
    for nm, Suse in (('full S', S), ('in-band S only', Sb), ('1/f fast fill', Sf)):
        sims = []
        for d in range(a.draws):
            t0 = time.time()
            Af = xspec.realise(Suse, idx, nframes, ref_frames=pad, seed=1000 + d)
            Aser = np.repeat(Af, save, axis=0)[:nframes * save] / save
            fr, _ = fl.run(c, ProfileDrive(c, P, Aser, 2e-4), p, nframes * save, save)
            U = bp.apply(units.smooth_frames(np.asarray(fr[:, cols], np.float64), kern),
                         fs, lo, hi)[t.burn:]
            del fr
            Z = U - U.mean(0, keepdims=True)
            Z = (Z / np.maximum(Z.std(0, keepdims=True), 1e-300)).T
            sims.append(float(t._prep(t.model_edges(Z=Z)[0]) @ t.y))
            print(f'    {nm:<16s} draw {d}: sim {sims[-1]:+.4f}   [{time.time()-t0:.0f}s]',
                  flush=True)
        out[nm] = sims
    print(f'\n  over {a.seconds:.0f}s, {a.draws} draws, all {t.nV} vertices:')
    for nm, v in out.items():
        print(f'    {nm:<16s} sim {np.mean(v):+.4f} +- {np.std(v):.4f}')
    base = np.mean(out['full S'])
    for nm in ('in-band S only', '1/f fast fill'):
        print(f'    {nm:<16s} vs full S: {np.mean(out[nm]) - base:+.4f}')


if __name__ == '__main__':
    main()
