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

```bash
python best_fit.py --oversample 4 --decay-s 9.03 --spread-mm-s 3 --bold-smooth \
  --pad 4096 --impulse-frames 224 --iters 400 --val-vert 0 --draws 3 \
  --regions subcortical --split 40
```

**+0.6679 ± 0.0033** Spearman over 2M edges, 577 s realisation, Moran gap 0.064, field
rank 29.4. Sensory at matched channel count gives +0.5695 ± 0.0162.

`--val-vert 0` matters: without it, `best_fit` turns on early stopping against a held-out
COVARIANCE, which stopped the sensory arm at step 5 of 400 while subcortical ran to 399.
Use `--select-iters` (realised held-out score) or disable it, never leave it on by default.

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

## 3. Ruled out

Each of these was measured, and each is a reason not to spend effort there.

| candidate | measurement |
|---|---|
| **field rank** | best rank-12 approximation of the target reaches Spearman 0.973; rank-5 reaches 0.940 |
| **reachable span** | 0.952 at 400 directions on the retuned medium (0.960 old clock) |
| **target reliability** | ceiling +0.9679; per-vertex mean 0.9218 |
| **restricting the fit to reliable vertices** | +0.0056 ± 0.0030 on held-out reliable vertices (1.8 sd), −0.045 on all vertices |
| **structural connectivity** | monotone worse: +0.5695 → 0.4953 for λ = 0 → 3 |
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
indistinguishable. Max-entropy is the WORST despite the highest rank, which kills the idea
that "field rank tracks the score" carries from stopping points to the family.

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

## 4. Open

1. **The edge distribution.** The model saturates: the binned mean tracks the diagonal to
   empirical FC ≈ 0.2 then flattens, returning ~+0.28 where the data has +0.6. It is also
   over-dispersed overall (edge sd ratio 1.44 sensory, 1.60 subcortical) and shows a
   population of false-positive strong edges. `plot_edges.py`. Not itemised anywhere in the
   ceiling budget — this is the live question.
2. **Data-driven splitting.** `split_parcels` bisects the MESH graph by Fiedler vector, so
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

## 5. Debugging lessons

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

**Report scatter, not means.** A "real trade" was claimed from 2 draws with no spread
measured; the FC cost turned out to be 1.8 sd. Separately, a ±0.016 scatter was carried
across from a different configuration where the true value was ±0.005.

---

## 6. Things to keep reporting

Multi-draw mean **and** scatter; solve correlation (Pearson and Spearman vs the raw
target); Moran gap; field rank; and the realisation length in SECONDS as well as frames.
On the fMRI clock, quote 577 s (3,578 frames at TR/4) so it matches the data's duration.

Field rank has a reference now: a single fMRI run measures **10.9–13.4**. The group FC's
90.2 is an average over 99 subjects and is NOT what one simulated run should match —
`fc_score.attach_rank`'s docstring cites 95 for "the empirical FC" and that is the wrong
comparison.

---

## 7. Theory

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

## 8. Code map

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
