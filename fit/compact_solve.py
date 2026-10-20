"""Replace a large solve's full cross-spectrum with its factor, in place.

Factored solves written since this tool existed store `S_L`/`S_tr` and drop the full
(nf, K, K) array above 512 channels (see xspec.load_S). Files written before that hold the
full array - 7.6 GB at 1,880 channels. This recovers a factor from it, one bin at a time,
by eigendecomposition: S_f = A_f A_f^H with A_f = U sqrt(ev) over the nonzero eigenvalues,
which for a rank-20 solve is 20 columns. The result is stored as S_L (zero-padded to the
largest rank across bins) with S_tr = 1, so xspec.load_S rebuilds S as S_L S_L^H.

NOT bit-identical, unlike a factor saved by the solve itself: the rebuilt S matches the
stored one to rounding, and the reconstruction error is printed and written into the file
(`S_compact_err`, max |dS| / max |S|). Draws made from the compacted file differ from
draws made from the original at that level. The original is replaced only if the error is
below --tol.

  python fit/compact_solve.py mf60_v1880 flat1880_r20 ...
"""
import _path  # noqa: F401
import os, time, argparse
import numpy as np

from paths import RESULTS


def factor_of(S):
    nf, K = S.shape[0], S.shape[1]
    As, rmax = [], 0
    for f in range(nf):
        M = 0.5 * (S[f] + S[f].conj().T)
        ev, U = np.linalg.eigh(M)
        k = ev > max(float(ev.max()), 0.0) * 1e-12
        A = (U[:, k] * np.sqrt(ev[k])) if k.any() else np.zeros((K, 0), complex)
        As.append(A); rmax = max(rmax, A.shape[1])
    L = np.zeros((nf, K, rmax), complex)
    for f, A in enumerate(As):
        L[f, :, :A.shape[1]] = A
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--tol', type=float, default=1e-9)
    ap.add_argument('--force', action='store_true', help='compact even at <= 512 channels')
    a = ap.parse_args()
    for tag in a.tags:
        t0 = time.time()
        f = os.path.join(RESULTS, f'xspec_{tag}.npz')
        z = np.load(f, allow_pickle=True)
        if 'S' not in z.files:
            print(f'  {tag}: no full S stored, nothing to do'); continue
        if z['S'].shape[1] <= 512 and not a.force:
            print(f'  {tag}: {z["S"].shape[1]} channels - small enough to keep whole'); continue
        S = z['S']
        L = factor_of(S)
        Sr = L @ np.conj(L).transpose(0, 2, 1)
        err = float(np.abs(Sr - S).max() / max(np.abs(S).max(), 1e-300))
        del Sr
        if err > a.tol:
            print(f'  {tag}: reconstruction error {err:.2e} above {a.tol:g}; left as is'); continue
        d = {k: z[k] for k in z.files if k != 'S'}
        d.update(S_L=L, S_tr=1.0, S_compact_err=err,
                 S_compact_note=np.array('full S replaced by an eigendecomposition factor '
                                         'by fit/compact_solve.py; rebuilds to rounding'))
        before = os.path.getsize(f)
        tmp = f + '.tmp.npz'
        np.savez(tmp, **d); os.replace(tmp, f)
        print(f'  {tag}: S {S.shape} -> factor {L.shape}, error {err:.1e}, '
              f'{before/2**30:.2f} GB -> {os.path.getsize(f)/2**20:.0f} MB  '
              f'[{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
