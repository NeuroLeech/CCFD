"""Hand-tune the medium: graded wave speed and damping, with the input structure fixed.

The input is whatever the last search found - loaded from its pickle and held constant -
so anything that moves is the medium. Speed and damping are log-linear in the z-scored
cortical maps, c(x) = c0 * exp(sum_j a_j m_j(x)) and sig(x) = sig0 * exp(sum_j b_j m_j(x)),
which grades them smoothly over the whole sheet instead of stepping at region borders.

Reported per run: similarity, Moran gap, effective rank of the field, and the penalised
total, so a gain in one can be seen against a loss in another.

  python play_fluid.py                                    # the frozen medium, as a datum
  python play_fluid.py --a myelin=0.3,thickness=-0.2      # faster where myelinated
  python play_fluid.py --sweep myelin --target speed      # scan one coefficient
  python play_fluid.py --a myelin=0.3 --figure            # draw the fields it implies
"""
import os, pickle, glob, argparse
import numpy as np

from mesh_cache import load_cortex
from fc_score import FCTarget
from fc_moran import MoranMatch
from paths import RESULTS
import fluid as fl
import ladder

MAPS = fl.MAPS_DEFAULT
GLOBALS = {                     # name -> (default scan range, log spacing)
    "c0":   ((0.3, 3.0), True),      # global wave speed
    "Ld":   ((8.0, 300.0), True),    # rotation: f = c / Ld, small Ld = strong Coriolis
    "sig0": ((1e-4, 1e-1), True),    # global damping rate
}


def parse_coefs(text, maps=MAPS):
    """'myelin=0.3,sulc=-0.1' -> array aligned with `maps`."""
    a = np.zeros(len(maps))
    if not text:
        return a
    for part in text.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        k = k.strip()
        if k not in maps:
            raise SystemExit(f"unknown map {k!r}; known: {', '.join(maps)}")
        a[maps.index(k)] = float(v)
    return a


def load_input(pkl=None):
    """Ladder parameters from a search pickle, fluid part discarded."""
    pkl = pkl or sorted(glob.glob(os.path.join(RESULTS, "cma_fc", "cma_ladder_*.pkl")),
                        key=os.path.getmtime)[-1]
    d = pickle.load(open(pkl, "rb"))
    x = d["best"][1]
    if d.get("fluid"):
        x = x[:-fl.N_PARAM]
    return x, d, os.path.basename(pkl)


def evaluate(cortex, xl, d, p, target, mm, seed=90_000, tapers=None, D=None,
             override=None):
    c, _ = fl.fields(cortex, p)
    dt = fl.CFL * cortex.d.min() / float(c.max())
    drive = ladder.make_drive(cortex, xl, d["nsteps"], dt, amp=2e-4,
                              rung=d.get("rung", 4), seed=seed, tapers=tapers,
                              regions=d["regions"], D=D, override=override)
    frames, _ = fl.run(cortex, drive, p, d["nsteps"], 25, d.get("sponge", True))
    Z, _ = target.model_z(frames)
    sim = float(target._prep(target.model_edges(Z=Z)[0]) @ target.y)
    gap = mm.gap(Z)
    rank = target.effective_rank(frames[target.burn:])
    total = sim - 2.0 * max(0.0, gap - 0.10) - 1.0 * max(0.0, (10.0 - rank) / 10.0)
    return dict(sim=sim, gap=gap, rank=rank, total=total, frames=frames, c=c,
                drive=drive, dt=dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkl", default=None, help="search pickle supplying the input")
    ap.add_argument("--c0", type=float, default=1.0)
    ap.add_argument("--Ld", type=float, default=52.4)
    ap.add_argument("--sig0", type=float, default=1e-3)
    ap.add_argument("--a", default="", help="speed coefficients, e.g. myelin=0.3,sulc=-0.1")
    ap.add_argument("--b", default="", help="damping coefficients, same form")
    ap.add_argument("--sweep", default=None,
                    help="what to scan: a map (myelin/thickness/sulc, with --target), "
                         "a global fluid term (c0/Ld/sig0), or a ladder parameter")
    ap.add_argument("--target", default="speed", choices=("speed", "damp"),
                    help="for a map sweep: which field its coefficient enters")
    ap.add_argument("--range", nargs=2, type=float, default=None, dest="rng",
                    help="scan range; defaults per quantity (log-spaced for the globals)")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--seed", type=int, default=90_000)
    ap.add_argument("--input", default="", dest="inp",
                    help="override ladder parameters, e.g. coherence=0.3,p_sat=0.8")
    ap.add_argument("--sweep-input", default=None, dest="sweep_input",
                    help="ladder parameter to scan instead of a map coefficient")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--save", default=None, help="save the field of the last run")
    ap.add_argument("--video", nargs="?", const="play", default=None, metavar="TAG",
                    help="render the field as an mp4 (results/videos/TAG_field.mp4)")
    ap.add_argument("--video-drive", action="store_true", dest="video_drive",
                    help="also render the injected drive, same projection and cadence")
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()

    cortex = load_cortex("fsaverage5", verbose=False)
    target = FCTarget(cortex, verbose=False)
    mm = MoranMatch(cortex, target)
    xl, d, name = load_input(args.pkl)
    from input2 import parcel_tapers
    tapers = parcel_tapers(cortex, verbose=False)
    D, _ = ladder.parcel_geodesic(cortex, d["regions"], verbose=False)
    print(f"input from {name} (held fixed), maps: {', '.join(MAPS)}")

    base = dict(mode="maps", maps=MAPS, c0=args.c0, Ld=args.Ld, sig0=args.sig0,
                a=parse_coefs(args.a), b=parse_coefs(args.b))
    override = {}
    for part in filter(None, (q.strip() for q in args.inp.split(","))):
        k, v = part.split("="); override[k.strip()] = float(v)
    if override:
        print(f"  input overrides: {override}")

    def show(tag, p, ov=None):
        r = evaluate(cortex, xl, d, p, target, mm, args.seed, tapers, D,
                     ov if ov is not None else (override or None))
        print(f"  {tag:34s} sim {r['sim']:+.4f}  gap {r['gap']:.3f}  "
              f"rank {r['rank']:5.1f}  total {r['total']:+.4f}  "
              f"(speed {r['c'].min():.2f}-{r['c'].max():.2f})")
        return r

    def scan(lo, hi, log=False):
        return (np.geomspace(lo, hi, args.n) if log else np.linspace(lo, hi, args.n))

    name = args.sweep or args.sweep_input
    if name:
        ladder_names = set(ladder.decode(np.full(ladder.N_PARAM, 0.5)))
        if name in MAPS:
            j, key = MAPS.index(name), ("a" if args.target == "speed" else "b")
            lo, hi = args.rng or (-0.45, 0.45)
            print(f"  scanning {args.target} coefficient for {name}:")
            for v in scan(lo, hi):
                p = dict(base); p[key] = base[key].copy(); p[key][j] = v
                show(f"{name} {args.target} {v:+.2f}", p)
        elif name in GLOBALS:
            (dlo, dhi), log = GLOBALS[name]
            lo, hi = args.rng or (dlo, dhi)
            what = {"c0": "global wave speed", "Ld": "rotation (f = c/Ld)",
                    "sig0": "global damping"}[name]
            print(f"  scanning {what}:")
            for v in scan(lo, hi, log):
                p = dict(base); p[name] = float(v)
                show(f"{name} {v:.4g}", p)
        elif name in ladder_names:
            lo, hi = args.rng or (0.0, 1.0)
            print(f"  scanning input parameter {name}:")
            for v in scan(lo, hi):
                ov = dict(override); ov[name] = float(v)
                show(f"{name} {v:+.3g}", base, ov)
        else:
            raise SystemExit(f"unknown quantity {name!r}; sweep one of: "
                             f"{', '.join(MAPS)} (with --target), "
                             f"{', '.join(GLOBALS)}, or a ladder parameter "
                             f"({', '.join(sorted(ladder_names))})")
        return

    r = show(f"c0 {args.c0:g} Ld {args.Ld:g} sig0 {args.sig0:g}", base)
    if args.save:
        np.save(args.save, r["frames"])
        print(f"  wrote {args.save}")
    if args.figure or args.video:
        from render_regimes import _proj
        proj = _proj(cortex.V, cortex.F)
    if args.figure:
        from surface_plots import fluid_maps
        fluid_maps(cortex, base, proj, os.path.join(RESULTS, "fluid_maps_play.png"))
    if args.video:
        from surface_plots import movie
        from paths import VIDEOS
        tag, dr = args.video, r["drive"]
        note = (f"c0 {args.c0:g} Ld {args.Ld:g} sig0 {args.sig0:g}"
                f"  sim {r['sim']:+.3f} gap {r['gap']:.3f} rank {r['rank']:.1f}")
        movie(r["frames"], cortex, proj, dr.Aser, 25, r["dt"],
              os.path.join(VIDEOS, f"{tag}_field.mp4"), f"field   {note}", fps=args.fps)
        if args.video_drive:
            # the injected source at the same saved steps, rebuilt from the drive itself
            src = (dr.Aser[::25][:len(r["frames"])].astype(np.float32)
                   @ dr.P.astype(np.float32))
            movie(src, cortex, proj, dr.Aser, 25, r["dt"],
                  os.path.join(VIDEOS, f"{tag}_drive.mp4"), f"drive   {note}",
                  fps=args.fps)


if __name__ == "__main__":
    main()
