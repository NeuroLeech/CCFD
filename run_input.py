"""Run one simulation with the input specified by hand, and describe what came out.

explore.py samples random genomes and ga.py searches over them; neither lets you
ask "what does driving V1 and 10r together actually do?". This does. Everything
the genome holds - regions, loadings, sparsity, timescales - and the three fluid
parameters it holds FROZEN (Ld, sponge strength, sponge width) are exposed as
flags, defaulting to the frozen regime so that a bare run is the same fluid every
other part of the pipeline uses.

The measures are computed by explore.run_genome, unchanged, so numbers printed
here are directly comparable with the GA's and with the stage-1 sweep's.

    # drive V1 and 10r on one shared latent, both positive
    python run_input.py --regions V1,10r

    # two latents: V1 alone on the first, 10r and 10v in antiphase on the second
    python run_input.py --regions V1,10r,10v --loadings "1,0; 0,1; 0,-1" \
                        --tau 20,40 --silent 0.8,0.9

    # same input, a much less rotational fluid, and a video
    python run_input.py --regions V1 --Ld 200 --video

    # which parcels exist
    python run_input.py --find front
"""
import sys, os, json, time, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from paths import FIELDS, VIDEOS
import genome as G


# ------------------------------------------------------------------ region names
def _short(name):
    """'L_V1_ROI' -> 'V1'."""
    if name.startswith("L_") and name.endswith("_ROI"):
        return name[2:-4]
    return name


def _lut(c):
    d = {}
    for pid in c.parcels:
        nm = c.names[int(pid)]
        d[_short(nm).lower()] = int(pid)
        d[nm.lower()] = int(pid)
    return d


def resolve_regions(spec, c):
    """'V1,10r' or '1,65' -> [1, 65]. Names are case-insensitive and the L_/_ROI
    wrapper is optional. An unknown name lists the nearest ones rather than
    failing silently on a typo."""
    lut = _lut(c)
    out = []
    for tok in str(spec).split(","):
        t = tok.strip()
        if not t:
            continue
        if t.lstrip("+-").isdigit():
            pid = int(t)
            if pid not in set(int(x) for x in c.parcels):
                raise SystemExit(f"parcel id {pid} is not in the atlas (1-180)")
        elif t.lower() in lut:
            pid = lut[t.lower()]
        else:
            shorts = [_short(c.names[int(p)]) for p in c.parcels]
            near = ([s for s in shorts if t.lower() in s.lower()]
                    or difflib.get_close_matches(t, shorts, n=8, cutoff=0.4))
            raise SystemExit(f"unknown region {t!r}. did you mean: "
                             + (", ".join(near) if near else "(nothing close)")
                             + "\n  python run_input.py --find <substring>  lists parcels")
        if pid in out:
            raise SystemExit(f"region {t!r} given twice")
        out.append(pid)
    if not out:
        raise SystemExit("--regions is empty")
    return out


def find_regions(sub, c):
    sub = sub.lower()
    rows = [(int(p), _short(c.names[int(p)]), int((c.lab == p).sum()))
            for p in c.parcels]
    hit = [r for r in rows if sub in r[1].lower()] if sub not in ("", "all") else rows
    print(f"  {'id':>4}  {'name':<12} {'vertices':>8}")
    for pid, nm, n in hit:
        print(f"  {pid:4d}  {nm:<12} {n:8d}")
    print(f"  {len(hit)} of {len(rows)} parcels")


# ------------------------------------------------------------------ the loadings
def parse_loadings(spec, K):
    """Rows are regions, columns latents. ';' separates rows, ',' columns.

    Omitted, every region loads +1 on a single shared latent - the simplest thing
    that is not the trivial one-region case, and what the stage-1 sweep used."""
    if spec is None:
        return np.ones((K, 1))
    rows = [r for r in str(spec).replace("\n", ";").split(";") if r.strip()]
    if len(rows) == 1 and K > 1 and len(rows[0].split(",")) == K:
        # a flat list of K numbers: one latent, one loading per region
        return np.array([float(v) for v in rows[0].split(",")], float).reshape(K, 1)
    L = np.array([[float(v) for v in r.split(",")] for r in rows], float)
    if L.shape[0] != K:
        raise SystemExit(f"--loadings has {L.shape[0]} rows for {K} regions")
    return L


def parse_vec(spec, r, name, lo=None, hi=None):
    """One value broadcast to every latent, or one per latent."""
    v = np.array([float(x) for x in str(spec).split(",")], float)
    if len(v) == 1:
        v = np.repeat(v, r)
    if len(v) != r:
        raise SystemExit(f"--{name} has {len(v)} values for {r} latents")
    if lo is not None and (v.min() < lo or v.max() > hi):
        raise SystemExit(f"--{name} outside [{lo}, {hi}]: {v}")
    return v


# ------------------------------------------------------------------ reporting
def report(res, c, p, rig):
    if not res["ok"]:
        print(f"\n  RUN FAILED: {res['reason']}")
        return
    L = np.asarray(p["loadings"])
    print(f"\n  input")
    print(f"    {'region':<12} {'verts':>6}  loadings")
    for k, pid in enumerate(p["regions"]):
        print(f"    {_short(c.names[int(pid)]):<12} {int((c.lab == pid).sum()):6d}  "
              + "  ".join(f"{v:+.2f}" for v in L[k]))
    print(f"    latent tau      " + "  ".join(f"{t:6.1f}" for t in p["taus"]))
    print(f"    latent silent   " + "  ".join(f"{s:6.2f}" for s in p["silents"])
          + f"   (realised " + ", ".join(f"{s:.2f}" for s in res["silent"]) + ")")
    print(f"    amp {p['amp']:.3g}  balance {p.get('balance', 'temporal')}  "
          f"mass residual {res['mass_residual']:.2e}")

    print(f"\n  fluid")
    print(f"    Ld {p['Ld']:.1f} mm  (L/Ld {228/p['Ld']:.1f})   "
          f"sponge {p['sponge_strength']:.2f} over {p['sponge_width']:.0f} mm")

    lin = "yes" if res["linear_ok"] else "NO - outside the linear regime"
    print(f"\n  amplitude")
    print(f"    peak {100*res['peak_frac']:.3f}% of layer depth   linear? {lin}")
    print(f"    rms  {100*res['rms_frac']:.4f}%   active fraction {res['active_frac']:.3f}")
    print(f"    energy drift x{res['energy_drift']:.2f} over the run")

    print(f"\n  spatial")
    print(f"    length scale     {res['length_scale']:8.2f} mm")
    print(f"    anisotropy       {res['anisotropy']:8.3f}")
    print(f"    orient coherence {res['orient_coherence']:8.3f}   "
          f"dir coherence {res['dir_coherence']:.3f}")

    print(f"\n  temporal")
    print(f"    wave speed       {res['wave_speed']:8.3f}")
    print(f"    pattern tau      {res['pattern_tau']:8.1f}")
    print(f"    field decorr     {res['decorr_time']:8.1f}   drive decorr "
          f"{res['drive_tau']:.1f}   ratio {res['tau_ratio']:.2f}")
    print(f"    var modulation   {res['var_mod']:8.3f}   "
          f"(low = travelling, high = standing)")
    print(f"\n  ran in {res['seconds']:.1f}s")


def fmri_scores(res, rig):
    """The two GA measures, on the same footing as ga.py computes them."""
    from stage2 import _normalise, smooth_decimate, similarity, richness, fmri_pcs, scaled
    Pm = smooth_decimate(_normalise(res["parcels"]), factor=rig.f["decimate"])
    sim, rich = similarity(Pm, fmri_pcs()), richness(Pm)
    s, r = scaled(sim, rich)
    print(f"\n  vs fMRI   ({Pm.shape[1]} model frames; scaled: 0 = white noise, "
          f"1 = the brain)")
    print(f"    similarity {sim:.3f}  (scaled {s:+.2f})")
    print(f"    richness   {rich:.3f}  (scaled {r:+.2f})")
    return dict(similarity=sim, richness=rich, sim_scaled=s, rich_scaled=r)


# ------------------------------------------------------------------ video
def render(res, c, path, title, labels=None, clip=None, fps=None):
    """One row of three views plus the drive trace. Same projection as
    render_regimes so a manual run looks like the swept ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from render_regimes import _proj, CLIP, FPS

    Hs, dtf = res["frames"], res["dt_frame"]
    proj = _proj(c.V, c.F)
    # the driven parcel usually holds the largest amplitudes by a wide margin, so
    # a percentile below ~96 is what makes the rest of the surface visible at all
    vl = float(np.percentile(np.abs(Hs), clip if clip is not None else CLIP))
    fps = fps or FPS
    fig = plt.figure(figsize=(4.0*len(proj), 4.8))
    gs = fig.add_gridspec(2, len(proj), height_ratios=[1.0, 0.33],
                          top=0.86, bottom=0.10, hspace=0.08, wspace=0.02)
    scs = []
    for k, (xy, vis, nm) in enumerate(proj):
        ax = fig.add_subplot(gs[0, k])
        scs.append(ax.scatter(xy[vis, 0], xy[vis, 1], c=np.zeros(vis.sum()),
                              s=3.0, linewidths=0, cmap="RdBu_r",
                              vmin=-vl, vmax=vl))
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(nm, fontsize=10)

    axd = fig.add_subplot(gs[1, :])
    d = res["drive_series"][:len(Hs)]
    tvec = np.arange(len(Hs))*dtf
    labels = labels or [f"region {j+1}" for j in range(d.shape[1])]
    for j in range(d.shape[1]):
        axd.plot(tvec, d[:, j]/max(np.abs(d).max(), 1e-30), lw=0.9,
                 label=labels[j])
    axd.set_xlim(tvec[0], tvec[-1]); axd.set_ylim(-1.15, 1.15)
    axd.set_yticks([]); axd.set_xlabel("time units", fontsize=9)
    axd.set_ylabel("drive", fontsize=9); axd.tick_params(labelsize=8)
    if d.shape[1] <= 8:
        axd.legend(fontsize=7, ncol=min(d.shape[1], 4), loc="upper right",
                   framealpha=0.7)
    cursor = axd.axvline(tvec[0], color="k", lw=1.2)
    fig.suptitle(title + f"    scale ±{vl:.1e}", fontsize=11, y=0.985)
    txt = fig.text(0.5, 0.905, "", ha="center", fontsize=10, family="monospace")

    def upd(i):
        for sc, (_, vis, _) in zip(scs, proj):
            sc.set_array(Hs[i][vis])
        cursor.set_xdata([tvec[i], tvec[i]])
        txt.set_text(f"t = {tvec[i]:7.1f}")
        return []

    animation.FuncAnimation(fig, upd, frames=len(Hs), blit=False).save(
        path, writer=animation.FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)


# ------------------------------------------------------------------ main
def build_parser():
    import argparse
    ap = argparse.ArgumentParser(
        description="Run the fluid model on a hand-specified input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--regions", type=str,
                    help="comma-separated parcel names or ids, e.g. V1,10r,10v")
    ap.add_argument("--find", type=str,
                    help="list parcels whose name contains this, then exit "
                         "('all' for every parcel)")
    # input
    ap.add_argument("--loadings", type=str, default=None,
                    help="rows=regions, cols=latents, ';' between rows. "
                         "default: every region +1 on one shared latent")
    ap.add_argument("--tau", type=str, default="20",
                    help="latent timescale(s) in time units; one value or one per latent")
    ap.add_argument("--silent", type=str, default="0.8",
                    help="fraction of the run with no drive, per latent")
    ap.add_argument("--amp", type=float, default=G.AMP_FIXED)
    ap.add_argument("--balance", choices=("temporal", "spatial"), default="temporal",
                    help="temporal: no net mass across the run. spatial: none at "
                         "any step, and impossible with a single region")
    ap.add_argument("--erode-to", type=int, default=0, metavar="N",
                    help="shrink every driven region's taper to about N vertices, "
                         "so region identity is not confounded with region size")
    # fluid (frozen in the genome; free here)
    ap.add_argument("--Ld", type=float, default=G.LD_FIXED,
                    help="deformation radius in mm; surface is 228 mm across")
    ap.add_argument("--sponge-strength", type=float, default=G.SPONGE_STRENGTH_FIXED)
    ap.add_argument("--sponge-width", type=float, default=G.SPONGE_WIDTH_FIXED)
    # run
    ap.add_argument("--nsteps", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=0, help="drive noise realisation")
    ap.add_argument("--mesh", type=str, default="fsaverage5")
    # output
    ap.add_argument("--fmri", action="store_true",
                    help="also score similarity and richness against the MSC fMRI")
    ap.add_argument("--save", type=str, default=None, metavar="NAME",
                    help=f"write the field to {FIELDS}/NAME.npz")
    ap.add_argument("--video", action="store_true",
                    help=f"render an mp4 into {VIDEOS}/manual")
    ap.add_argument("--clip", type=float, default=96.0,
                    help="colour scale percentile of |h|; lower for more contrast "
                         "outside the driven region")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--frame-every", type=int, default=50, metavar="N",
                    help="simulation steps per rendered frame. this is also the "
                         "sampling the wave measures run on, so changing it moves "
                         "anisotropy/speed/pattern_tau a little")
    ap.add_argument("--name", type=str, default=None,
                    help="label for saved files; default is built from the regions")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    from mesh_cache import load_cortex
    from explore import Rig, run_genome

    if a.find is not None:
        find_regions(a.find, load_cortex(a.mesh, verbose=False))
        return 0
    if not a.regions:
        build_parser().print_help()
        return 2

    rig = Rig(a.mesh, fluid=dict(nsteps=a.nsteps, wave_every=a.frame_every))
    c = rig.c
    regions = resolve_regions(a.regions, c)
    L = parse_loadings(a.loadings, len(regions))
    r = L.shape[1]
    p = dict(regions=np.array(regions, int), loadings=L,
             silents=parse_vec(a.silent, r, "silent", 0.0, 0.999),
             taus=parse_vec(a.tau, r, "tau", 1e-3, 1e6),
             amp=float(a.amp), balance=a.balance,
             Ld=float(a.Ld), sponge_strength=float(a.sponge_strength),
             sponge_width=float(a.sponge_width))
    if a.balance == "spatial" and len(regions) == 1:
        raise SystemExit("spatial balance annihilates a single region's drive "
                         "(see input_model.NetworkDrive) - use --balance temporal")

    if a.erode_to:
        from sweep_fields import eroded_taper
        T, ids = rig.tapers
        T = T.copy()
        pos = {int(pp): i for i, pp in enumerate(ids)}
        for pid in regions:
            tap, n, lvl = eroded_taper(c, pid, a.erode_to)
            T[pos[pid]] = tap
            print(f"  eroded {_short(c.names[pid])}: {int((c.lab == pid).sum())} "
                  f"-> {n} vertices (depth >= {lvl})")
        rig.tapers = (T, ids)

    name = a.name or ("+".join(_short(c.names[pid]) for pid in regions)
                      + f"_Ld{a.Ld:.0f}_s{a.seed}")
    need_field = bool(a.save or a.video)
    print(f"\n  {c}")
    print(f"  dt {rig.dt:.4f} x {a.nsteps} steps = {rig.duration():.0f} time units "
          f"(wave crossing ~200)")

    t0 = time.time()
    res = run_genome(None, rig, seed=a.seed, params=p, keep_field=need_field)
    report(res, c, p, rig)
    if not res["ok"]:
        return 1
    scores = fmri_scores(res, rig) if a.fmri else {}

    meta = dict(name=name, regions=[int(x) for x in regions],
                region_names=[_short(c.names[pid]) for pid in regions],
                loadings=L.tolist(), taus=p["taus"].tolist(),
                silents=p["silents"].tolist(), amp=p["amp"], balance=a.balance,
                Ld=p["Ld"], sponge_strength=p["sponge_strength"],
                sponge_width=p["sponge_width"], nsteps=a.nsteps, seed=a.seed,
                erode_to=a.erode_to, mesh=a.mesh,
                **{k: (float(res[k]) if np.isscalar(res[k]) else res[k])
                   for k in ("peak_frac", "rms_frac", "length_scale", "anisotropy",
                             "wave_speed", "pattern_tau", "decorr_time",
                             "tau_ratio", "var_mod", "dir_coherence",
                             "orient_coherence", "active_frac")},
                **scores)
    if a.save:
        path = os.path.join(FIELDS, f"{a.save}.npz")
        np.savez(path, H=res["frames"], drive=res["drive_series"],
                 dt_frame=res["dt_frame"], regions=np.array(regions),
                 parcels=res["parcels"], meta=json.dumps(meta))
        print(f"  wrote {path}  ({os.path.getsize(path)/1e6:.0f} MB, "
              f"{len(res['frames'])} frames)")
    if a.video:
        os.makedirs(os.path.join(VIDEOS, "manual"), exist_ok=True)
        path = os.path.join(VIDEOS, "manual", f"{name}.mp4")
        title = (f"{', '.join(meta['region_names'])}   Ld {a.Ld:.0f} "
                 f"(L/Ld {228/a.Ld:.1f})   tau {np.round(p['taus'], 0)}   "
                 f"silent {np.round(p['silents'], 2)}")
        render(res, c, path, title, labels=meta["region_names"],
               clip=a.clip, fps=a.fps)
        print(f"  wrote {path}")
    print(f"  total {time.time()-t0:.0f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
