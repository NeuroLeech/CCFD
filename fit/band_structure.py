"""Does the solve SHAPE the input outside the passband, or leave it at the white start?

This is the question the realised Spearman cannot answer. The passband is how the target is
measured, not the band the medium runs in, so a solve that puts structure only inside
0.01-0.08 Hz has been told nothing about the rest of the spectrum - and for the LINEAR
observable it cannot be, because an LTI medium moves no power between frequencies and H is
~0 above the passband, so the objective's gradient there is exactly zero. The envelope's
square couples pairs of frequencies, which is what gives those bins a gradient at all.

Three scale-free readings of S per frequency band, all on the solved cross-spectrum with no
simulation involved:

  offdiag/diag    mean |S_ab| off the diagonal over mean S_aa. 0 is the white identity the
                  solve starts from; larger means the channels are coupled
  participation   (sum lambda)^2 / sum lambda^2 of S, averaged over bins - the effective
                  number of input modes. 100 is full rank, i.e. untouched white
  % power         share of sum_f w_f tr(S_f) above the passband

  python fit/band_structure.py
"""
import _path  # noqa: F401
import os, argparse
import numpy as np

from paths import RESULTS
import xspec


def read(tag, kind):
    f = os.path.join(RESULTS, {'xspec': 'xspec_', 'envfit': 'envfit_',
                               'corrfit': ''}[kind] + tag + '.npz')
    z = np.load(f, allow_pickle=True)
    S = xspec.load_S(z)
    w = np.asarray(z['H_w' if 'H_w' in z.files else 'w'], float)
    idx = np.asarray(z['idx'], np.int64)
    pad = int(z['pad']) if 'pad' in z.files else 4096
    fs = float(z['frame_s']) if 'frame_s' in z.files else 0.16125
    band = tuple(float(v) for v in z['band']) if 'band' in z.files else (0.01, 0.08)
    return S, w, idx, pad, fs, band


def stats(S, w, sel):
    if not sel.any():
        return np.nan, np.nan
    off, dia, pr = [], [], []
    for f in np.flatnonzero(sel):
        A = np.abs(S[f])
        d = np.diag(A)
        off.append((A.sum() - d.sum()) / (len(d) * (len(d) - 1)))
        dia.append(d.mean())
        ev = np.clip(np.linalg.eigvalsh(0.5 * (S[f] + S[f].conj().T)).real, 0, None)
        pr.append((ev.sum() ** 2) / max((ev ** 2).sum(), 1e-300))
    return float(np.mean(off) / max(np.mean(dia), 1e-300)), float(np.mean(pr))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--rows', default=','.join([
        'linear old:xspec:grclip400_nolag',
        'linear new:corrfit:corrfit_lin400_lam10_it400',
        'envelope old:envfit:env_grclip400',
        'envelope new:corrfit:corrfit_env400_lam0_it400',
        'envelope new 1000it:corrfit:corrfit_env400_lam0_it1000']))
    a = ap.parse_args()
    print(f'\n  {"":<22s} {"IN BAND":>19s}   {"ABOVE THE BAND":>19s}   {"% power":>8s}')
    print(f'  {"solve":<22s} {"offdiag/diag":>12s} {"modes":>6s}   '
          f'{"offdiag/diag":>12s} {"modes":>6s}   {"above":>8s}')
    for spec in a.rows.split(','):
        label, kind, tag = spec.split(':')
        try:
            S, w, idx, pad, fs, (lo, hi) = read(tag, kind)
        except FileNotFoundError:
            print(f'  {label:<22s} (no file for {tag})')
            continue
        f_hz = idx / (pad * fs)
        inb, above = (f_hz >= lo) & (f_hz <= hi), f_hz > hi
        tr = np.array([np.trace(S[f]).real for f in range(len(idx))]) * w
        o1, p1 = stats(S, w, inb)
        o2, p2 = stats(S, w, above)
        print(f'  {label:<22s} {o1:>12.4f} {p1:>6.1f}   {o2:>12.4f} {p2:>6.1f}   '
              f'{100*tr[above].sum()/max(tr.sum(),1e-300):>7.1f}%')
    print(f'\n  the white start the solves begin from would read 0.0000 / 100.0 in both '
          f'bands,\n  with 97.5% of its power above the passband')


if __name__ == '__main__':
    main()
