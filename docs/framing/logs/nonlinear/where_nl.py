"""Where in frequency the nonlinear field departs from the linear one. Draw 0 of the 30 s
rotation drive: the nonlinear fields saved at 30x, 35x, 40x, and one linear run of the same
drive (linear fields scale exactly with amplitude, so one serves all three). Per frequency
range: the power of (nonlinear - linear) as a share of the linear field's power, and where
the difference power sits. The same on the scored observable (1.29 s kernel + 6 mm)."""
import os, sys
ROOT = "/home/user/CCFD"
sys.path[:0] = [os.path.join(ROOT, d) for d in ("core", "fit", "targets", "analysis")]
os.chdir(ROOT)
import numpy as np
import torch_fit, xspec, fluid as fl, units, timescale, surface_smooth, fc_score
from mesh_cache import load_cortex
from paths import RESULTS
tag = "nobp_sm6_s3_rot30_box"
c = load_cortex("fsaverage5", verbose=False)
ref = torch_fit.load_reference(tag)
t = fc_score.FCTarget(c, fc_path=ref["fc_path"], centre="double", verbose=False)
lin = torch_fit.Fit(ref, c, t, 577.0, device="cpu", amp=6e-3, nl_flux=0.0, nl_adv=0.0)
Aser = np.load(os.path.join(RESULTS, f"drive_{tag}_nl30.npy"))
d = xspec.ProfileDrive(c, ref["P"], Aser, 6e-3)
d.Aser = Aser
Flin30, _ = fl.run(c, d, lin.p, len(Aser), lin.save)
burn, fs = 50, ref["frame_s"]
kern = units.smoothing_kernel(timescale.bold_fwhm_frames(fs, verbose=False), verbose=False)
K = surface_smooth.smoothing_matrix(c, 6.0)
cols = np.asarray(t.cols)
edges = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.775, 1 / (2 * fs)]

def spec(X):
    X = np.asarray(X[burn:], np.float64); X = X - X.mean(0)
    return (np.abs(np.fft.rfft(X, axis=0)) ** 2).mean(1)

def obs(X):
    return surface_smooth.smooth(units.smooth_frames(np.asarray(X, np.float32), kern), K)[:, cols]
for m in (30, 35, 40):
    Fnl = np.load(os.path.join(RESULTS, f"frames_{tag}_nl{m}_raw.npy")).astype(np.float64)
    n = min(len(Fnl), len(Flin30))
    Fl = Flin30[:n] * (m / 30.0)
    D = Fnl[:n] - Fl
    f = np.fft.rfftfreq(n - burn, fs)
    for nm, (A, B) in (("bare field", (Fl, D)), ("scored observable", (obs(Fl), obs(D)))):
        Pl, Pd = spec(A), spec(B)
        tot = Pd[1:].sum() / Pl[1:].sum()
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = (f > lo) & (f <= hi)
            rows.append(f"{lo:g}-{hi:.3g} Hz: {Pd[k].sum()/max(Pl[k].sum(),1e-300):6.1%} "
                        f"(holds {Pd[k].sum()/Pd[1:].sum():5.1%} of it)")
        print(f"x{m} {nm}: nonlinear part = {tot:.1%} of the linear power overall")
        for r in rows:
            print("    " + r)
