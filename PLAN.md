# CCFD — where the project is

A rotating shallow-water field on the cortical surface, driven at named Glasser parcels,
fitted to resting-state FC by SOLVING for the input cross-spectrum rather than searching
for it. The medium is linear, so the field covariance depends on the drive only through
its cross-spectral density `S(f)`, and matching FC is convex over one Hermitian PSD matrix
per frequency.

Numbers here are on the **fMRI clock** unless marked *(old clock)*. The two are not
comparable: everything before the retune ran ~99x too fast and was scored over 29 s of
simulated time against FC computed from 577 s of data.

---

## Current best

**Read section 0 first.** As of 2026-09-04 the default target is the RBC cohort, not
nilearn's release, so this command no longer scores what is recorded below — it gives
+0.7102 ± 0.0005 against the new target. Under the passband the data was filtered to it
gives +0.4701, and the swept optimum there is a different medium entirely: spread
1.47 mm/s, decay 25 s, +0.6325. Everything in this section and in sections 1-9 was
measured against the nilearn target and remains comparable with itself.

```bash
python best_fit.py --oversample 4 --decay-s 9.03 --spread-mm-s 6 --bold-smooth \
  --pad 4096 --impulse-frames 224 --iters 400 --val-vert 0 --draws 2 \
  --regions subcortical --split 40 --profile taper
```

**+0.7204 ± 0.0009** Spearman over 2M edges, 577 s realisation, Moran gap 0.060, field
rank 22.1; 47 pieces driving 8,542 mm². The same configuration at `--spread-mm-s 3` gives
+0.6679 ± 0.0033, and sensory at matched channel count and spread 3 gives +0.5695 ± 0.0162.
Spread was swept and 6 mm/s is where that sweep peaked (reach = spread × decay = 53 mm).

`--val-vert 0` matters: without it, `best_fit` turns on early stopping against a held-out
COVARIANCE, which stopped the sensory arm at step 5 of 400 while subcortical ran to 399.
Use `--select-iters` (realised held-out score) or disable it, never leave it on by default.

**Rerunning this command today gives +0.7197 ± 0.0010** (gap 0.060, rank 22.4), because
the solve was restructured on 2026-09-03 — see below. The +0.7204 ± 0.0009 above is what
the PRE-CHANGE solve produced, and it still reproduces: the old `xspec.py` was swapped
back in on 2026-09-03 and returned +0.7204 ± 0.0009, gap 0.060, rank 22.1, matching the
recorded line exactly. Every score elsewhere in this file predates the change and came
from that path, so they remain comparable with each other.

### The solve, restructured (2026-09-03)

`xspec.solve` evaluates the same objective by different arithmetic:

- `_project` now returns the PSD factor `L` it already computes, so the covariance is
  `sum_f Re(B_f B_f^H)` with `B_f = sqrt(2 w_f) H_f L_f`. Stacked over frequency and split
  into real and imaginary parts that is two REAL gemms, `Zr^T Zr + Zi^T Zi`, in place of
  nf complex triple products — a quarter of the real flops, in one large call.
- the adjoint keeps the real `M` out of complex arithmetic.
- `adjoint(Ctn)` is hoisted out of the loop. `Ctn` never changes, so this part is exact.

`solve(ref=True)` restores the per-frequency loops. The two agree to 1e-15 at 30
iterations on every branch — plain, `nblock=2`, `share`, `psd=False`, `freq_keep`, `spec`,
held-out — and diverge to 6.5e-4 in the objective and 1.2e-2 in `S` by iteration 400, with
`corr(C_fast, C_ref) = 0.999986`. The objective never converges and never stalls, so a
rounding-level difference has 400 accepted steps to grow. That drift is the whole of the
0.0007; nothing else about the run changed.

`--workers` now defaults to `min(12, cpu_count)` and covers the extra draws as well as the
impulses. Both pools were checked bit-identical against their serial paths, and the draw
pool returns the same sim/gap/rank to four decimals. One machine, warm cache:

| stage | before | after |
|---|---|---|
| solve, 400 iterations at nf 135 | 472 s | 86 s |
| whole command | 11:50 | 2:51 |
| impulse stage, COLD cache, 47 pieces | 126 s | 19 s |

The impulse cache key does not record whether the responses came from the pool, which is
why that path is checked for bit-identity rather than for agreement.

---

## 0. The RBC target and the passband (2026-09-04)

The target is no longer nilearn's release. It is 100 RBC subjects' `rest-645` runs
(`fc_group_rbc.py`), so the target and the task scans are the same people, one session,
fMRIPrep 24.1.1 + XCP-D 0.10.6, and one resampling into fsaverage5 — the same
`fc_vertexwise.resample` call the MSC data goes through. 9,310 vertices against 9,217;
against the old target on the 9,216 they share, pearson +0.927 and spearman +0.889, with
the new edges larger (sd 0.1311 against 0.1033).

**XCP-D bandpasses everything at 0.01–0.08 Hz, order 2.** Both derivatives carry it in
`SoftwareFilters`. So the anchors below do NOT transfer: re-measured on RBC they read 1/e
4.06 s and slope −0.90, but a passband inside the measurement window makes those
properties of the filter. `autocorr.py --source rbc` prints them with that caveat.

The passband is therefore taken as given rather than re-anchored. `bandpass.py` applies
the same filter to the model's observable — `--bandpass 0.01,0.08` multiplies H by |H|²
and filters the realised frames — so what is compared is what survives the filter on both
sides, not BOLD's temporal statistics, which the model has more of and the data has lost.

| against the RBC target | sim | gap | rank |
|---|---|---|---|
| old config, no bandpass | +0.7102 ± 0.0005 | 0.062 | 24.2 |
| old config, bandpassed | +0.4701 ± 0.0070 | 0.165 | 4.4 |
| bandpassed, pad 16384 (84 usable bins, not 37) | +0.4718 ± 0.0018 | 0.158 | 4.7 |

The target swap costs little. The passband costs 0.24, and not for want of frequency
parameters: doubling the usable bins lands inside the draw scatter.

### Why the rank collapses, and what it costs

The unfiltered model's dimensionality lives above the passband. Restricting one run to
successive bands:

| band (Hz) | rank | FC r=0.5 at |
|---|---|---|
| 0.01–0.08 | 4.3 | 60.4 mm |
| 0.08–0.2 | 8.4 | 30.8 mm |
| 0.2–0.4 | 15.3 | 15.6 mm |
| 0.4–0.775 | 35.6 | 9.0 mm |

Rank rises with frequency and spatial scale falls with it — the dispersion relation of a
wave medium, where a low-frequency window is a large-wavelength window and a sheet holds
few long-wavelength modes. The bandpass selects the coarse end of the field.

**Decay is not identifiable from the filtered autocorrelation.** Exponentially-correlated
noise through this filter maps a 60× range of input decay (1–60 s) onto 4.29–7.70 s out,
and the empirical 4.06 s is below that whole range. The score is the usable handle, so
both spread and decay were swept against it — decay is no longer pinned at 9.03 s.

### The sweep, 40 runs (`bp_sweep.py`)

sim over spread × decay, all bandpassed, 400 iterations, 2 draws:

|spread|d3|d6|d9|d12|d15|d20|d25|d35|d50|d70|d100|
|---|---|---|---|---|---|---|---|---|---|---|---|
|1.11| | |+0.528| | | | | | | | |
|1.47|+0.479|+0.556|+0.577|+0.593|+0.604|+0.614|**+0.633**|+0.608|+0.621|+0.597|+0.626|
|1.84|+0.497|+0.567|+0.593|+0.604|+0.627|+0.624|+0.618|+0.617|+0.609|+0.617| |
|2.58|+0.534| | |+0.604|+0.609|+0.612|+0.617| | | | |
|2.95| | |+0.573|+0.591|+0.587|+0.594|+0.586| | | | |
|4.42| | |+0.535| | | | | | | | |
|5.89| | |+0.470| | | | | | | | |
|8.84| | |+0.402| | | | | | | | |
|12.16| | |+0.123| | | | | | | | |

Highest is **+0.6325 ± 0.0163 at spread 1.47 mm/s, decay 25 s** (`grid_s1.5_d25`), against
+0.4701 for the old 5.89 / 9.03 configuration under the same filter.

- **Spread is the sharper axis.** 2.95 is below every other row at every decay by margins
  outside scatter; 5.89 and above fall away steeply.
- **sim is flat in decay above ~25 s** — 1.47 runs +0.597 to +0.633 across a 4× range,
  non-monotonically, with per-run scatter up to 0.016. The maximum is not separated from
  the plateau.
- **The product is not sufficient.** At matched reach, spread 2.95 × decay 9.03 (26.6 mm)
  gives +0.5730 and spread 1.84 × decay 15 (27.6 mm) gives +0.6192 — 3× the scatter apart.
  `reach = spread × decay` does not summarise the pair under this objective.
- **Moran gap and rank keep moving where sim does not.** Along spread 1.47 the gap falls
  0.073 → 0.054 → 0.047 → 0.041 from decay 25 to 100, the lowest recorded anywhere here
  including the unfiltered +0.7102 run's 0.062, while rank climbs 15.1 → 20.8 past the
  empirical 13.0. Two configurations 4× apart in decay score the same and differ 1.8× in
  gap.
- **The FC correlation length is set by spread, not decay**: 22–23 mm across the whole
  1.47 row, 33–38 mm across 2.95, against an empirical **9.1 mm**. Reducing spread closes
  part of that and the score falls before it closes.

Empirical comparison values, on the cohort's own resting runs and so already bandpassed:
rank **13.0**, FC r=0.5 at **9.1 mm** (20 subjects).

`bp_sweep.py` reads the runs from `results/` rather than the logs, recovering spread and
decay from each saved `x`, and detects from the realisation's spectrum whether the
observable was filtered — an unfiltered control caught by the same glob otherwise tops the
ranking for the wrong reason. Its sim is draw 0 only, where the logs report the draw mean.

---

## 1. The clock (`timescale.py`)

`c0` is not a speed knob and never was: `dt = CFL·d_min/c` means distance per step is
`CFL·d_min` whatever `c` is (verified identical at c0 = 0.25, 1, 4). `save` sets distance
per frame. Seconds per frame is a free anchor, and it had been set two incompatible ways:

- `units.py` declared a 300 mm/s spread → one frame = 6.54 ms
- `bo_step` matched model-frame FC to TR-sampled FC → one frame = one TR

They differ by 99x. Under the first, a 1,120-frame window is 7.3 s and its lowest
frequency is 0.137 Hz — above the entire resting-state band, so restricting the solver to
what BOLD sees was not even expressible.

**Both anchors are now measured from NKI:**

| quantity | value |
|---|---|
| BOLD spectrum | `f^-2.60`; 84% of variance in 0.01–0.1 Hz, 66% in 0.01–0.03 |
| BOLD autocorrelation | 9.03 s at 1/e, 14.9 s integrated → 38.8 independent samples per 577 s |
| single-run field effective rank | 10.9–13.4 |
| group FC effective rank | 90.2 |

Decay is pinned to the data; spread is the one free parameter, with reach = spread × decay.
At TR/4: frame 0.161 s, `save` 8, damping 2.23e-3, 3,578 frames = 577 s.

**Frequency resolution matters more than the medium did.** 0.01–0.03 Hz holds 63% of BOLD
power and a 165 s window resolves three bins there. Padding 1024 → 4096 frames (661 s,
0.0015 Hz bins) moved the score +0.291 → +0.504 with no model change. Zero-padding is free
because the impulse has decayed by frame 224.

---

## 2. What works

### Subcortically-driven input, +0.098 over sensory

17 parcels grouped by thalamic origin (`subparcels.SUBCORTICAL`; SMA → 6mp, AI → AAIC by
hand). All arms solved identically, 400 accepted steps, 3 draws:

| arm | solve pearson | realised | Moran gap | field rank |
|---|---|---|---|---|
| sensory, 47 pieces | 0.6893 | +0.5695 ± 0.0162 | 0.082 | 15.2 |
| **subcortical, 47 pieces** | 0.7601 | **+0.6679 ± 0.0033** | 0.064 | 29.4 |
| subcortical, 58 pieces | 0.7717 | **+0.6743 ± 0.0049** | 0.044 | 34.5 |

About 6 sd at matched channel count, and matched driven area (1,448 vs 1,532 vertices), so
it is placement not amount. The Moran gap improves alongside.

Per parcel, the worst-fitting prefrontal regions move furthest: 9-46d +0.285 → +0.688,
46 +0.337 → +0.715, 25 +0.441 → +0.791. **But the gain is not local**: undriven cortex
gains +0.088 against driven cortex's +0.097, so this is a better input basis for the whole
sheet, not a repair of regions that lacked drive. EC (−0.044) and PIT (−0.032) get WORSE
despite now being driven.

### Lagged covariance constrains what FC cannot see (`lagged.py`, `xspec.solve_lagged`)

With `M_f = H_f S_f H_fᴴ` Hermitian, `Re(M)` symmetric and `Im(M)` antisymmetric:

```
Φ(τ) = Σ_f w_f · 2 · [ cos θ · Re(M_f) + sin θ · Im(M_f) ]      θ = 2πfτ
```

At τ = 0 the sine vanishes, so the standard objective constrains only `Re(M_f)` and leaves
the phase structure free. The antisymmetric blocks are orthogonal to the symmetric part.

The data carries it: group lagged covariance, global signal removed, antisym/sym = 0.046 at
1.9 s, 0.141 at 5.2 s, 0.269 at 7.7 s; CSD phase consistency 1.9–2.2× chance.

**No weight avoids the rank collapse.** 3 draws each, `ns` = inside 2 sd:

| member | FC pearson | FC spearman | lag 1.9 s | lag 5.2 s | rank |
|---|---|---|---|---|---|
| zero-lag | **+0.6428** | **+0.5537** | +0.1656 | +0.1173 | **18.7** |
| wa=0.3 | +0.6318 ns | +0.5337 | +0.1543 ns | +0.1022 ns | 6.7 |
| wa=1 | +0.6343 ns | +0.5348 | +0.1762 ns | +0.1551 | 6.8 |
| wa=3 | +0.6264 | +0.5252 | +0.2115 | +0.2160 | 7.1 |
| wa=10 | +0.5914 | +0.4887 | **+0.2762** | **+0.3264** | 11.1 |

Rank falls 18.7 → 6.7 at the SMALLEST weight and stays near 7 through wa=3; wa=0.3 is
strictly dominated. From wa=1 up the trade is monotone: ~1 unit of Spearman for 2–3 of lag.

**To fix before reuse:** `lagged.empirical` applies GSR by default, so the objective's
zero-lag block is a GSR'd target while the score is not.

---

## 3. Measured, with the conditions that were tested

Each row is a measurement at the configuration named, not a general claim. A null here
means the effect did not appear THERE — at that parameter range, that region set, that
clock — and several entries have already changed sign when the clock changed. Read the
conditions before reusing any of them as a reason to skip something.

| candidate | measurement |
|---|---|
| **field rank** | best rank-12 approximation of the target reaches Spearman 0.973; rank-5 reaches 0.940 |
| **reachable span** | 0.952 at 400 directions on the retuned medium (0.960 old clock) |
| **target reliability** | ceiling +0.9679; per-vertex mean 0.9218 |
| **restricting the fit to reliable vertices** | +0.0056 ± 0.0030 on held-out reliable vertices (1.8 sd), −0.045 on all vertices |
| **structural connectivity, long-range** | monotone worse: +0.5695 → 0.4953 for λ = 0 → 3 *(old clock)*; on the fMRI clock, +0.7204 uncoupled → +0.7162 at λ=0.30 |
| **structural connectivity, incl. short fibres** | `--coupling-mm 10 --coupling-keep 1.0`: +0.7117 / +0.7011 / +0.6788 at λ = 0.02 / 0.04 / 0.08. At matched perturbation strength (dt·bound 0.0139 vs 0.0136) short+long is +0.6788 against long-only's +0.7162. **The per-edge-length breakdown was not run**, so whether the 10–30 mm band moved is unmeasured |
| **unsupervised denoising** | PCA saturates at K=50 and never beats no truncation; bandpass −0.033 and GSR −0.008 on split-half reliability |
| **MSC as an alternative dataset** | see below |
| **whitening** | +0.006 old clock, **−0.219** on the fMRI clock (rank 15.2 → 6.8) |

Notes on two of them.

**Structural connectivity.** Coupling opens 0 → 35 reachable directions carrying 9.5% of
the uncoupled residual, but the field's leading six modes stay 0.90–0.96 unchanged except
at λ=3 where the fit has already lost 0.07. The earlier λ ≤ 0.03 screen tested a term too
weak to do anything (the field moves 0.8% at λ=0.003, 7.4% at 0.03, 45% at 0.3): choose a
parameter range by measuring what it does, not from precedent. An old-clock λ=0.3 gain of
+0.011 did NOT survive the retune.

**MSC.** Two causes, only one real. `fc_vertexwise` states "the MSC data already carries
6 mm FWHM from its own pipeline" and passes `smooth_fwhm=0`; measured, it carries none
(nearest-neighbour 0.308 mean / 0.090 median against NKI's 0.910 / 0.926). The resampling
was innocent — resampling the source sphere's own coordinates reproduces the target's to a
median 0.05°. Smoothing recovers what it can, but at 14 mm where local smoothness MATCHES
NKI, mid-range FC is still zero (−0.007 at 20–40 mm against NKI's +0.103). That is the
signature of aggressive cleaning (GSR + nuisance regression + bandpass in the MSC
pipeline), and choosing between pipelines has no ground truth to appeal to. What it would
have bought is on record: ceiling 0.95–0.98 against NKI's 0.783, one session (0.756)
beating a whole NKI run (0.446). **Do not redo** the smoothness finding, the resample
validation or the smoothing sweep.

### FC does not determine the input *(old clock)*

Members of the admissible family, each held within eps = 0.01 of the argmax, realised and
scored:

| member | realised | field rank | pieces used |
|---|---|---|---|
| argmax | +0.6317 | 116.7 | 29.6 |
| min-rank | +0.6301 | 85.2 | **1.9** |
| local coalitions | +0.6322 | 120.5 | 33.4 |
| coordinated | +0.6291 | 95.7 | 4.7 |
| independent | +0.6244 | 140.2 | 38.6 |
| max-entropy | +0.6183 | 158.6 | 41.3 |

An input through effectively **1.9 of 47 pieces** scores as well as one spread over 29.6.
Realised score spans 0.0139 while spatial concentration spans **22×**. Four of the six are
indistinguishable. Max-entropy is the WORST despite the highest rank, so "field rank
tracks the score" — which holds across stopping points — does not carry to the family.

### The gap, decomposed *(old clock, `ceiling.py`)*

| rung | value | step |
|---|---|---|
| perfect | +1.0000 | |
| target reliability | +0.9679 | −0.0321 target noise |
| span ceiling (400 dims) | +0.9604 | −0.0075 architecture |
| solve objective reached | +0.7478 | −0.2126 |
| Spearman vs raw | +0.7380 | −0.0098 surrogate objective |
| simulated, solve vertices | +0.6643 | −0.0738 fidelity |
| simulated, held out | +0.6459 | −0.0184 generalisation |
| reported | +0.6418 | −0.0040 sampling |

Dropping the PSD cone (Hermitian only, a diagnostic not a model) reaches +0.8356, so at
least 41% of the 0.2126 is the constraint and not the solver. The robust conclusion is the
NEGATIVE: neither architecture (0.0075) nor target noise (0.0321) can account for the gap.

---

## 4. The 10-30 mm band

Accuracy split by the geodesic distance between an edge's endpoints has a minimum at
10-30 mm. It is the deepest feature in the fit and it has not moved under anything tried.

### The curve, across input parameterisations

Taper vs fixed-width Gaussian kernels, subcortical, spread 6, 47 pieces (`/tmp/prof.sh`):

| config | area driven | realised | 0-10 | 10-20 | 20-30 | 30-40 | 40-60 | 60-80 | 80-120 | 120-250 |
|---|---|---|---|---|---|---|---|---|---|---|
| taper | 8,542 mm² | +0.7204 ± 0.0009 | 0.635 | 0.530 | **0.512** | 0.628 | 0.722 | 0.714 | 0.696 | 0.685 |
| gauss 10 masked | 8,394 | +0.7199 ± 0.0024 | 0.640 | **0.518** | 0.524 | 0.638 | 0.723 | 0.712 | 0.689 | 0.685 |
| gauss 10 unmasked | 16,793 | +0.7256 ± 0.0003 | 0.639 | 0.529 | **0.536** | 0.650 | 0.729 | 0.720 | 0.698 | 0.698 |
| gauss 16 unmasked | 29,620 | +0.7307 ± 0.0023 | 0.613 | 0.555 | **0.525** | 0.646 | 0.733 | 0.720 | 0.706 | 0.704 |

At matched area, profile SHAPE moves the score by less than the draw scatter. Coverage
lifts the tails (120-250 mm: 0.685 → 0.704) without filling the notch; widening to
29,620 mm² is a soft version of switching the input prior off, for +0.010.

The notch also survived piece diameter 9-22 mm, medium reach 14-54 mm, and the coupling
term (section 3). Five parameterisations spanning input geometry, input extent, medium
transport and an added anatomical term, with the band's position unchanged in all of them.

### It is not a scoring artefact (`band_ceiling.py`)

Per-vertex band accuracy is a Spearman over that vertex's own annulus — median 120
partners — so the obvious worry is that there is too little spread within an annulus to
rank. Measured against the same estimator as the target: 99 subjects split in half, group
FC per half, full-matrix double centring, then read out on the band.

| | value |
|---|---|
| band ceiling (half A vs half B), mean | **+0.9119** |
| model band accuracy, mean | +0.5793 |
| shortfall | **+0.3326** |
| corr(model band accuracy, ceiling) | **−0.010** |
| corr(model band accuracy, target sd within annulus) | +0.111 |
| corr(model band accuracy, per-vertex reliability) | +0.074 |

Where the data is reliable has no relationship to where the model fails. The failing
parcels have high ceilings — VVC 0.954, area 1 0.938, OFC 0.930, VMV3 0.917, LBelt 0.910,
A4 0.889 — giving shortfalls up to +1.02.

**Nuisance controls have to be the right statistic.** `band_fail.py` first tested mean
|target FC| in the band (−0.021) and called the artefact unlikely; the quantity a rank
correlation over an annulus actually depends on is the SPREAD within it, and that needed
the split-half run to settle.

### Where it fails (`band_fail.py`)

All 9,217 vertices, each scored on its own 10-20 mm annulus, taper run.

- Band accuracy mean +0.5793 against all-partner +0.6952, and the two correlate at only
  **+0.270** — a different spatial pattern, not a projection of the global one.
- Worst AT the drive and improving away from it: +0.535 (0-1 mm from the nearest driven
  vertex), +0.554, +0.569, +0.604 (10-20 mm), +0.593, +0.580, +0.614. The all-partner
  curve is flat across the same bins.
- Model |FC| in the band is ~2x the target's at every distance (0.46-0.54 vs 0.25-0.30).
- Worst parcels, with their all-partner accuracy alongside: VVC −0.068/+0.489,
  A4 −0.006/+0.516, 23d +0.031/+0.298, LBelt +0.123/+0.665, PBelt +0.140/+0.614,
  VMV3 +0.159/+0.582, 25 +0.243/+0.813, OFC +0.247/+0.460, EC +0.256/+0.360,
  V8 +0.259/+0.450, area 1 +0.286/+0.697, OP4 +0.305/+0.780.
- Best: 7AL +0.870, STGa +0.869, 9a +0.850, AIP +0.846, PGi +0.839, 7PC, 6d, LIPv, PGs,
  PHT, PF, PFm — parietal and lateral prefrontal.
- Grouped by the input parcel a vertex sits NEAREST: OFC +0.412, 25 +0.415, A1 +0.450,
  EC +0.459, 3b +0.463, V1 +0.485 at the bottom; 7AL +0.742, 9-46d +0.718, 6a +0.714,
  IP1 +0.710, 46 +0.656 at the top. Primary sensory and orbitofrontal/limbic drives are
  the bad neighbourhoods, association drives the good ones.

The parcel list clusters where a primary area abuts its belt — A1/A4/LBelt/PBelt,
V1/V8/VVC/VMV3, 3b/area 1/OP4.

### Local over-smoothing does not explain it (`band_homog.py`)

Local homogeneity = mean profile correlation with the vertex's band partners, model and
target measured the same way over 2,000 common partners.

| | target | model | excess |
|---|---|---|---|
| 10-20 mm homogeneity | +0.6887 | +0.7618 | +0.0731 |

The model IS smoother on average, and that excess does not locate the failure:
corr(band accuracy, excess) **+0.068**. The distance profile runs the other way — excess
−0.016 at the drive where the fit is worst, rising monotonically to +0.138 at 30-40 mm
where it is fine. At the failing parcels the sign is mostly negative (LBelt −0.114,
V8 −0.091, PBelt −0.083, OFC −0.074), while two of the best-fitting carry the largest
excesses (STGa +0.190, PHT +0.169).

What does align, weakly, is target homogeneity: the failing parcels sit at 0.83-0.89 and
the best at 0.50-0.68, corr −0.124. Not pursued further.

### The mesh is not the limit (`spatial_scale.py`, `mesh_check.py`, `mesh_transfer.py`)

The impulse-response fields fall to r = 0.5 over **7.6 mm**, under three fsaverage5 vertex
spacings (mean nearest-neighbour 2.90 mm), which is close enough to the grid to be worth a
convergence check.

The same CONTINUUM medium `p` was integrated on fsaverage6 — every rate in `p` is per unit
time, so the finer mesh takes its own smaller timestep from the CFL bound and the physics
per second is unchanged; `save` 16 → 40 holds the frame at 0.1613 s (matched to +0.07%).
fsaverage5's vertices nest exactly inside fsaverage6 (9,373 shared, white coordinates
agreeing to 1.4 µm), so readout needs no interpolation; the drive is carried the other way
by nearest neighbour. 20 min for 47 pieces.

| | fs5 | fs6 |
|---|---|---|
| autocorrelation at 10-15 mm | +0.270 | +0.272 |
| at 15-20 mm | +0.143 | +0.149 |
| r = 0.5 at | 7.62 mm | 7.51 mm |
| rms amplitude | 1.000 | 1.003 |
| transfer function \|corr\|, 0.01-0.1 Hz | — | **0.973** (phase < 3°, gain within 1.3%) |

Four times the vertices produces no finer spatial structure, and the transfer function the
solve consumes is unchanged in the band. The 7.6 mm scale is the physics of this medium at
spread 6, not the grid.

The time-domain responses DO diverge — +0.946 at one frame, +0.386 by the 9 s decay,
+0.029 at 36 s — but that is accumulated phase error above the band, and it is confounded:
the two inflated surfaces are separate inflations a median 3 mm apart, a ~6% path-length
difference over 50 mm, so refinement and metric cannot be separated by this run. The
autocorrelation and transfer-function results are statistics over the whole sheet and are
insensitive to that perturbation, which is why they carry the conclusion and the pointwise
correlation does not.

### What has not been tested here

- **Data-driven splitting** (section 5.2) — the one lever from the original list that
  changes WHICH vertices are grouped rather than the geometry of the grouping.
- **A connectivity term at piece scale.** The coupling operator acts on parcel means
  (median 18 mm), so it cannot represent anything below its own resolution — which is the
  scale of the band.
- **The short-fibre coupling arms broken out by edge length.** The scalars are in
  section 3; the per-band curves that would say whether they moved the notch were never
  computed.

---

## 5. Open

1. **The edge distribution.** The model saturates: the binned mean tracks the diagonal to
   empirical FC ≈ 0.2 then flattens, returning ~+0.28 where the data has +0.6. It is also
   over-dispersed overall (edge sd ratio 1.44 sensory, 1.60 subcortical) and shows a
   population of false-positive strong edges. `plot_edges.py`. Not itemised anywhere in the
   ceiling budget — this is the live question.
2. **Data-driven splitting.** The remaining untested lever on the 10-30 mm band
   (section 4), and the only one that changes which vertices are grouped rather than the
   geometry of the grouping. `split_parcels` bisects the MESH graph by Fiedler vector, so
   pieces are equal-area and follow geometry alone. Clustering vertices on their FC profile
   instead would let pieces follow functional boundaries; `xspec.medoid_subset` already
   does that clustering for a neighbouring purpose. Pairs naturally with the subcortical
   set — a piece straddling a functional boundary drives two things the solve cannot
   separate.
3. **Fidelity, 0.0738** *(old clock)*. The simulation does not reproduce its own
   prediction. `fidelity.py` measures it; nothing has tried to close it.
4. **HCP** if FIX-quality data is wanted — its surface timeseries are already FIX-cleaned,
   which is the one denoising route that does not need us to build a classifier.
5. **Concatenated segments** (`segments.py`). With segments differing in `H`, choosing
   lengths and inputs jointly is one convex solve over stacked `[H_1 … H_R]` with
   block-diagonal PSD — `xspec.solve(nblock=R)`. A speed library gave +0.5598 ± 0.0014
   against a single segment's +0.5695 ± 0.0162; the decay library +0.5844 ± 0.0110. Note
   `_project` assumes EQUAL block sizes, so libraries varying the driven set need NNLS.
   Admissibility is a sharper test than the score: `Σ_target ⪰ w₁Σ₁` caps the current
   model's contribution at **5%** of a concatenation, because its covariance has effective
   rank 25.6 against the target's 90.2.

---

## 6. Debugging lessons

Five errors this session were caught only by a check run AFTER a conclusion had been
reported. Each check was minutes of work.

**A gradient check cannot catch a convention error.** `lagged.phases` used
`exp(+2πifτ)` where `E[x(t)x(t+τ)ᴴ] = ∫P(f)e^{−2πifτ}df`. Model and adjoint agreed to
1e-9 — both described the transpose of the intended quantity, which flips the
ANTISYMMETRIC part exactly and leaves the symmetric part identical. It inverted a reported
conclusion ("the model propagates against the empirical lag gradient" — it propagates with
it). What caught it: comparing the PREDICTED lagged covariance against the SIMULATED one.
The wrong convention scores −0.78, the right one +0.78. **For anything with a time
direction, check the prediction against a simulation before reporting.**

**Do not duplicate an adjoint.** `solve_lagged` inlined its own copy with a separate sign
error on the `i`, which stalled the solve after two accepted steps of 400 while looking
like a result about lagged fitting. It now calls the finite-difference-checked
`lagged.model_lagged` / `adjoint_lagged` pair.

**Self-adjointness is not optional.** A non-symmetric double-centring that excluded the
diagonal from its means is not self-adjoint, so every gradient built on it silently lost a
term — a factor of 2.6. Use `Z(J·J)` with means over all entries; its adjoint is the same
operations in the other order.

**Check what a parameter does before choosing its range.** The λ ≤ 0.03 connectome screen
tested a term that moves the field by under 8%.

**Do not write verdicts.** A null at one configuration is a null AT that
configuration. Section 3 was headed "Ruled out" and several of its entries had already
changed sign across the clock retune. Record the measurement and the conditions; let the
reader decide what it forecloses.

**Report scatter, not means.** A "real trade" was claimed from 2 draws with no spread
measured; the FC cost turned out to be 1.8 sd. Separately, a ±0.016 scatter was carried
across from a different configuration where the true value was ±0.005.

---

## 7. Things to keep reporting

Multi-draw mean **and** scatter; solve correlation (Pearson and Spearman vs the raw
target); Moran gap; field rank; and the realisation length in SECONDS as well as frames.
On the fMRI clock, quote 577 s (3,578 frames at TR/4) so it matches the data's duration.

Field rank has a reference now: a single fMRI run measures **10.9–13.4**. The group FC's
90.2 is an average over 99 subjects and is NOT what one simulated run should match —
`fc_score.attach_rank`'s docstring cites 95 for "the empirical FC" and that is the wrong
comparison.

---

## 8. Theory

**Coalitions do not respect geometry.** Offset-vs-geodesic correlation is +0.39–0.45 at 19
whole parcels and collapses to +0.01–0.28 at 47 sensory pieces, +0.03–0.17 at spread. Not
a coverage effect — granularity. With 19 channels the medium's travel times pin the
phases; with 50+ the solve puts them anywhere. Amplitudes never correlated.

**The Moran gap tracks coverage** (0.064 at 16 pieces to 0.015 at 100) with no penalty
applied, and now improves with the subcortical set (0.082 → 0.044). Still unexplained.

**Rotation wants to be near zero.** It buys within-frequency pattern diversity (10 → 25
effective patterns) but collapses the span's participation rank 252 → 50, because waves get
trapped within a Rossby radius of their source.

**Prior art.** This is covariance completion, done in control theory and fluid dynamics:
Georgiou (2002) on which covariances a linear filter can produce; Hotz & Skelton (1987) on
covariance control; Zare, Jovanović & Georgiou on covariance completion, including "Colour
of turbulence" (JFM 812, 2017), which solves the same problem for linearised Navier–Stokes
with `minimise −log det(X) + γ‖Z‖_*` subject to `AX + XA* + Z = 0`. Two transferable ideas:
their objective is maximum ENTROPY of the OUTPUT covariance with a nuclear-norm penalty on
the FORCING (we do the reverse), and their white-noise-insufficiency test checks the
eigenvalue signs of `A·X_emp + X_emp·A*` with no fitting at all.

---

## 9. Code map

| file | role |
|---|---|
| **entry points** | |
| `best_fit.py` | reproduce the current best; `--regions`, `--oversample`, `--band` |
| `xspec.py` | transfer function, convex solve, family members, realisation, scoring |
| `timescale.py` | the fMRI clock: spectrum, autocorrelation, band, grid |
| `subparcels.py` | equal-area splitting; `region_set`, `SENSORY`, `SUBCORTICAL` |
| `fluid.py`, `swe_rot.py` | the medium and the integrator; `patch_field` for regional changes |
| **targets** | |
| `fc_score.py` | `FCTarget`: alignment, edge sample, Spearman or Pearson |
| `fc_group_nki.py` | the NKI group target (99 usable subjects) |
| `lagged.py` | empirical lagged covariance and the model's prediction of it |
| `reliability.py`, `holdout.py` | the ceiling, and the subject-split control |
| **diagnostics** | |
| `diag_maps.py` | where the fit fails, on the surface; reproducibility across halves |
| `plot_edges.py` | model vs empirical edge by edge |
| `ceiling.py` | the gap decomposed into named, measured steps |
| `diag_span.py`, `diag_residual.py` | reachable span; is the residual reachable |
| `vertex_quality.py`, `subject_fit.py` | per-vertex ceiling; single-subject rank and ceiling |
| `csd_probe.py` | is there consistent lead-lag in the data at all |
| `band_fail.py` | per-vertex accuracy inside one edge-length band; parcels, distance to drive, nearest input parcel |
| `band_ceiling.py` | split-half reliability of a vertex's band profile — the ceiling that band accuracy is scored against |
| `band_homog.py` | local homogeneity of model and target profiles over a band |
| `spatial_scale.py` | the spatial scale the fields actually carry, against mesh spacing |
| `mesh_check.py`, `mesh_transfer.py` | fsaverage5 vs fsaverage6 on the same continuum medium; fields and transfer function |
| **experiments** | |
| `family.py` | members of the admissible family; realisation laws; input bands |
| `lagfit.py` | lagged objective, realised and scored; `--wa` sweep |
| `segments.py` | concatenated segments with differing media; stacked block solve |
| `coupling_reach.py`, `connectome.py` | structural coupling, lags, span and modes |
| `solver_test.py` | is the solver or the model being measured |
| **closed** | |
| `denoise.py` | PCA/bandpass/GSR do not clean this data |
| `reliable_fit.py` | restricting the solve to reliable vertices does not help |
| `msc_reliability.py` | MSC — see section 3 |
| **pictures** | |
| `render_frames.py`, `surface_plots.py`, `plot_fc_map.py` | video, surface maps |
