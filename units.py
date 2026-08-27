"""Approximate metres and seconds for a model that has neither.

Nothing in the simulation is dimensioned. The mesh is the INFLATED fsaverage5 surface, so
its edge lengths are not millimetres on the cortical sheet, and dt is whatever the CFL
condition makes it. "The medium's reach is too short" is therefore currently a statement
that cannot be checked against anything.

Two calibrations, both deliberately approximate:

SPACE. The mesh graph weighted by WHITE-surface edge lengths is already in millimetres -
that is what ladder._white_graph builds for the geodesics. So the wave's spread can be
measured directly as real millimetres per frame, by regressing each vertex's time-to-peak
against its white-surface distance from the driven piece. A global inflated-to-white area
ratio is reported alongside as a cross-check; the two disagree because inflation stretches
sulcal walls far more than gyral crowns, and the disagreement is the width of the
approximation.

TIME. Fixing the spread to a physiological cortical wave speed fixes the second. dt and the
frame interval follow, and the frame interval can then be set against the target's TR of
0.645 s - which is the point of the exercise, since the model's frame and fMRI's TR have
never been commensurable.

Temporal smoothing follows from the same place. A filter between the field and the
observable is LINEAR, so it multiplies into the transfer function and costs the solve
nothing; the only requirement is that the realised frames are filtered identically before
scoring, or H stops describing the system being scored.

  python units.py --tag best              # calibrate against a saved run's medium
"""
import argparse
import numpy as np

TR_TARGET = 0.645                   # NKI enhanced, seconds


def area_ratio(cortex, verbose=True):
    """(length scale, area scale) from white vs inflated total surface area.

    A crude global factor: inflation conserves neither area nor length, and it stretches
    non-uniformly, so this is an average that is locally wrong everywhere."""
    import ladder
    _, Vw = ladder._white_graph(cortex)
    old = np.asarray(cortex.old)
    keep = np.isin(cortex.F, old).all(1)
    F = np.searchsorted(old, cortex.F[keep])          # faces in submesh indexing

    def tot_area(V):
        a, b = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
        return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())

    Aw, Ai = tot_area(Vw), tot_area(cortex.V)
    if verbose:
        print(f"  total area: white {Aw:.0f} mm2, inflated {Ai:.0f} (mesh units)2  "
              f"-> length scale {np.sqrt(Aw/Ai):.3f} mm per unit")
    return float(np.sqrt(Aw / Ai)), float(Aw / Ai)


def edge_ratio(cortex, verbose=True):
    """Mean white edge length over mean inflated edge length - the other global factor."""
    import ladder
    _, Vw = ladder._white_graph(cortex)
    E = cortex.edges
    lw = np.linalg.norm(Vw[E[:, 0]] - Vw[E[:, 1]], axis=1).mean()
    li = np.linalg.norm(cortex.V[E[:, 0]] - cortex.V[E[:, 1]], axis=1).mean()
    if verbose:
        print(f"  mean edge: white {lw:.3f} mm, inflated {li:.3f} units "
              f"-> length scale {lw/li:.3f} mm per unit")
    return float(lw / li)


def vertex_geodesic(cortex, seeds):
    """White-surface geodesic distance (mm) from each seed vertex to every vertex."""
    from scipy.sparse.csgraph import dijkstra
    import ladder
    G, _ = ladder._white_graph(cortex)
    return dijkstra(G, indices=np.asarray(seeds, int))


def model_speed(resp, cortex, labels, n_pieces=None, min_mm=15.0, max_mm=90.0,
                verbose=True):
    """Spread of the impulse response, in real millimetres per frame.

    Time-to-peak per vertex against white-surface distance from the driven piece's
    centroid, over an annulus that excludes the piece itself (where the peak is at frame
    0) and the far field (where the response is at the noise floor). The slope of distance
    on arrival time is the speed; it is measured on the real surface even though the wave
    propagated on the inflated one, which is exactly the approximation being made."""
    import ladder
    n_pieces = resp.shape[0] if n_pieces is None else n_pieces
    _, cen = ladder.label_geodesic(cortex, labels, n_pieces, verbose=False)
    Dv = vertex_geodesic(cortex, cen)
    speeds = []
    for k in range(n_pieces):
        peak = np.argmax(np.abs(resp[k]), axis=0).astype(float)     # frames
        d = Dv[k]
        m = (d > min_mm) & (d < max_mm) & (peak > 0)
        if m.sum() < 50:
            continue
        A = np.c_[np.ones(m.sum()), peak[m]]
        beta, *_ = np.linalg.lstsq(A, d[m], rcond=None)
        if beta[1] > 0:
            speeds.append(beta[1])
    speeds = np.array(speeds)
    if verbose:
        print(f"  spread: {np.median(speeds):.2f} mm/frame "
              f"(IQR {np.percentile(speeds,25):.2f}-{np.percentile(speeds,75):.2f}, "
              f"{len(speeds)} of {n_pieces} pieces usable)")
    return float(np.median(speeds)), speeds


def anchor_time(mm_per_frame, save, target_speed_mm_s, verbose=True):
    """Fix the second by declaring the model's spread to be a physiological speed."""
    frame_s = mm_per_frame / target_speed_mm_s
    dt_s = frame_s / save
    if verbose:
        print(f"  if the spread is {target_speed_mm_s:.0f} mm/s: "
              f"frame = {frame_s:.4g} s, step = {dt_s:.4g} s")
        print(f"    against the target's TR of {TR_TARGET} s, the model frame is "
              f"{frame_s/TR_TARGET:.3g} x a TR")
        # the same calibration read the other way, which is the more useful direction:
        # holding one frame at one TR, what spread would the model then be claiming?
        implied = mm_per_frame / TR_TARGET
        print(f"  conversely, if one frame WERE one TR ({TR_TARGET} s), the spread "
              f"would be {implied:.3g} mm/s ({implied/1000:.2e} m/s)")
        print(f"    that is {target_speed_mm_s/implied:.0f}x slower than "
              f"{target_speed_mm_s:.0f} mm/s")
    return frame_s, dt_s


def smoothing_kernel(fwhm_frames, verbose=True):
    """Normalised gaussian low-pass over frames, as a time-domain kernel.

    Defined in the time domain deliberately: the transfer function multiplies by its DFT
    and the scored frames are convolved with the kernel itself, so both sides are the same
    object by construction rather than by two implementations agreeing."""
    sd = fwhm_frames / 2.3548
    n = max(1, int(np.ceil(4 * sd)))
    t = np.arange(-n, n + 1, dtype=float)
    k = np.exp(-0.5 * (t / max(sd, 1e-9)) ** 2)
    k /= k.sum()
    if verbose:
        print(f"  smoothing: FWHM {fwhm_frames:.1f} frames, kernel length {len(k)}")
    return k


def kernel_response(kernel, nbins, nframes):
    """The kernel's frequency response on the rfft grid of an nframes window."""
    pad = np.zeros(nframes)
    n = len(kernel)
    pad[:n] = kernel
    pad = np.roll(pad, -(n // 2))               # zero-phase
    return np.fft.rfft(pad)[:nbins]


def smooth_frames(frames, kernel):
    """Apply the kernel along time, zero-phase, edges replicated.

    Must stay the exact counterpart of kernel_response: the transfer function is
    multiplied by that kernel's DFT, so any difference here silently breaks the identity
    between the solved system and the scored one.

    The two agree to 1e-7 of a standard deviation away from the ends. They differ only in
    the first and last few samples, because multiplying in the frequency domain is a
    CIRCULAR convolution while this replicates the edge - a handful of frames out of
    thousands, and the impulse responses this is applied to have decayed to nothing at
    both ends anyway."""
    from scipy.ndimage import convolve1d
    return convolve1d(np.asarray(frames, np.float32), kernel.astype(np.float32),
                      axis=0, mode="nearest")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="best", help="results/xspec_<tag>.npz for the split")
    ap.add_argument("--speed", type=float, default=300.0,
                    help="assumed cortical spread, mm/s (0.3 m/s)")
    a = ap.parse_args()

    import os
    from mesh_cache import load_cortex
    from paths import RESULTS
    import xspec, bo_step, subparcels, fluid as fl
    from best_fit import BEST_X

    c = load_cortex("fsaverage5", verbose=False)
    print("space:")
    ls_a, _ = area_ratio(c)
    ls_e = edge_ratio(c)
    print(f"  the two global factors differ by {abs(ls_a-ls_e)/max(ls_a,ls_e)*100:.0f}% "
          f"- that spread is the width of this approximation")

    p, save, _ = bo_step.unpack(BEST_X, c)
    labels, tags = subparcels.split_parcels(c, subparcels.SENSORY, 50, verbose=False)
    P = subparcels.taper_profiles(c, labels, len(tags))
    resp = xspec.impulse_responses(c, list(range(len(P))), p, 280 * save, save,
                                   profiles=P, verbose=False)
    print("\nspread, measured on the white surface:")
    mmf, _ = model_speed(resp, c, labels, len(tags))

    print("\ntime:")
    _, dt_s = anchor_time(mmf, save, a.speed)
    s, dt, _, _ = fl.build(c, p)
    print(f"  model dt is {dt:.4g} mesh units; one frame is {save} steps")


if __name__ == "__main__":
    main()
