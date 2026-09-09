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

**Read section 0+ first, then section 0.** As of 2026-09-04 the default target is the RBC cohort, not
nilearn's release, so this command no longer scores what is recorded below — it gives
+0.7102 ± 0.0005 against the new target. Under the passband the data was filtered to it
gives +0.4701, and the swept optimum there is a different medium entirely: spread
1.47 mm/s, decay 25 s, **+0.6901 ± 0.0018** at a 2,308 s realisation. Everything in this
section and in sections 1-9 was measured against the nilearn target at 577 s and remains
comparable with itself.

```bash
python fit/best_fit.py --oversample 4 --decay-s 9.03 --spread-mm-s 6 --bold-smooth \
  --pad 4096 --impulse-frames 224 --iters 400 --val-vert 0 --draws 2 \
  --regions subcortical --split 40 --profile taper
```

**+0.7204 ± 0.0009** Spearman over 2M edges, 577 s realisation, Moran gap 0.060, field
rank 22.1; 47 pieces driving 8,542 mm². The same configuration at `--spread-mm-s 3` gives
+0.6679 ± 0.0033, and sensory at matched channel count and spread 3 gives +0.5695 ± 0.0162.
~~Spread was swept and 6 mm/s is where that sweep peaked~~ (reach = spread × decay =
53 mm). **Superseded (§0):** that sweep ran against an unfiltered objective. Against the
bandpassed one the peak is at **1.47 mm/s**, and 5.89 scores +0.4772 there.

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

## 0+. 2026-09-08 — the basis, the span, the solver

**This section is newer than section 0 and supersedes it where they overlap.** Everything
here is the `g7_s1.5_d25` configuration unless said otherwise: RBC target, subcortical,
`--split 40`, taper, spread 1.47 mm/s, decay 25 s, bandpassed 0.01–0.08 Hz, `--bold-smooth`,
7-decay impulse window, pad 4096, 2,308 s, 2 draws. That command reproduced in session as
`sh_s40_n1` at **+0.6901 ± 0.0018, gap 0.065, rank 20.5**, matching the recorded line, so
everything below is measured against a baseline that reproduced.

**Nothing here beat it.** The best alternative tied within scatter at four times the
parameters. The incumbent command is still the incumbent.

### The input basis is one-hot, and the pieces are two rings deep

`taper_profiles` erodes `labels == i`, and the pieces are disjoint, so `P_k(v)` is nonzero
for exactly one `k`. Measured: max nonzero channels per vertex **1**, participation ratio
**1.00**. The solve sets 47 amplitudes and never a shape — the drive's spatial profile
inside a piece is frozen and identical for every input the model can express.

At `--split 40` a piece is **2–4 erosion rings deep (median 2)** against a median
equivalent diameter of 14.5 mm and 2.9 mm vertex spacing. With 2 rings the smoothstep takes
u ∈ {0.5, 1.0}, so the outer ring sits at exactly 0.500 of peak. It is a two-level step,
not a taper.

| driven area, 1,532 vertices / 8,542 mm² | share | mean total drive |
|---|---|---|
| on an internal seam (piece meets piece) | 34.5% | 0.394 |
| on the outer edge of the driven region | 32.4% | 0.404 |
| piece interior | 33.1% | 0.864 |

A third of the driven area sits at ~0.39 of peak because of where the atlas split fell, and
no input can lift it. `gauss_profiles`' docstring claim of 24.6% of touched area under half
the peak reproduces exactly.

**This reframes §4's profile null.** Channels seen per vertex: taper 1.00, gauss FWHM 6
1.05, gauss FWHM 10 masked 1.25, gauss FWHM 16 masked 1.71. The gauss-10 arm in §4's table
is still nearly one-hot, so "profile SHAPE moves the score by less than the draw scatter"
compared two bases that barely differ in overlap. It tested edge softness, not whether a
vertex can be driven by more than one channel.

Radial room by split, at constant 8,542 mm² footprint: `--split 40` → 47 pieces, 165 mm²,
14.5 mm, 2–4 rings; `--split 20` → 26 pieces, 19.7 mm, 2–6; `--split 12` → 18 pieces,
25.7 mm; `--split 8` and below → **17 pieces** (one per parcel), 510 mm², 25.5 mm, 2–6
rings, median 3.

### Concentric shells: within-piece shape, measured (`shell_profiles`, `--shells`)

`subparcels.shell_profiles` splits each piece into `nshell` shells by normalised erosion
depth using hat functions that partition unity in depth, so a piece's shells sum exactly to
the profile `taper_profiles` gives it and `--shells 1` is **bit-identical** to it. A vertex
is fed by up to two channels of its own piece, so the solve chooses the radial shape; core
and belt in antiphase is a centre-surround, which a one-hot basis cannot express at any
input. Same footprint throughout.

| tag | pieces | shells | channels | sim | gap | field rank | solve pearson |
|---|---|---|---|---|---|---|---|
| `sh_s40_n1` | 47 | 1 | 47 | **+0.6901 ± 0.0018** | 0.065 | 20.5 | +0.7564 |
| `sh_s1_n1` | 17 | 1 | 17 | +0.5802 ± 0.0041 | 0.107 | 16.8 | +0.6497 |
| `sh_s1_n2` | 17 | 2 | 34 | +0.6073 ± 0.0019 | 0.111 | 16.0 | +0.6680 |
| `sh_s1_n3` | 17 | 3 | 51 | +0.6138 ± 0.0017 | 0.103 | 16.1 | +0.6737 |
| `sh_s40_n2` | 47 | 2 | 94 | +0.6910 ± 0.0011 | 0.079 | 19.0 | +0.7549 |

- **At matched channel count**, 51 radial channels give +0.6138 against 47 tangential
  channels' +0.6901 — 0.076 apart, forty times the draw scatter.
- **Added to the incumbent**, 2 shells on the 47 pieces give +0.6910 ± 0.0011 against
  +0.6901 ± 0.0018: +0.0009, inside scatter, for 4× the parameters (135 × 94² ≈ 1.19M real
  against 298k). Gap moved the wrong way, 0.065 → 0.079; field rank 20.5 → 19.0.
- The solve's **own in-sample objective went down**, +0.7564 → +0.7549, with 4× the free
  parameters. The extra parameters were not consumed — 400 steps of a feasible-direction
  method spreading the same atoms over twice the channels.
- Within the shell arm returns fall off fast: 17 → 34 → 51 channels gives +0.5802 → +0.6073
  → +0.6138, so the second shell buys +0.027 and the third +0.0065.
- At `--split 40` a 2-ring piece cannot resolve 3 shells and collapses back toward one-hot:
  participation 1.00 → 1.73 at 2 shells → 1.33 at 3.

Position moved sim (+0.110 for 17 → 47 channels); within-piece shape did not (+0.0009).
Not tested: the in-plane dipole / directional half of the same idea, which is the same
class of change at fixed position.

### The span is not the resolution floor (`psf_g7_s1.5_d25.npz`)

Stacking `B_f = sqrt(2w_f) H_f` over frequency and orthonormalising gives `U`, whose
projector is the point-spread of every field the model can produce, for any `S`.
Conditions: this one used `nfreq 135` → 101 bins, where the saved runs use `nfreq 192` →
135 bins.

| retained directions | 5 | 10 | 20 | 47 | 100 | 200 | 400 | 600 |
|---|---|---|---|---|---|---|---|---|
| point spread, r = 0.5 (mm) | 33.1 | 28.2 | 24.8 | 21.1 | 16.0 | 11.3 | 7.8 | 6.1 |
| σ/σ₁ | 0.810 | 0.651 | 0.487 | 0.271 | 0.0908 | 0.0191 | 0.00275 | 0.000794 |

The basis crosses the data's 9.1 mm at ~350–400 directions, so **it can express the missing
scale**; the cost is gain, 10⁻⁴ to 10⁻⁶ in power. White input gives r=0.5 at 21.8 mm and
the realised field at 21.8 mm, both at the top-47 figure of 21.1.

The realised field already spreads **wider** than white input would — variance inside the
top-k directions is 0.2232 / 0.3680 / 0.6969 / 0.9368 / 0.9946 at k = 10 / 20 / 47 / 100 /
200, where white input gives 0.4381 / 0.6468 / 0.8827 / 0.9838 / 0.9990. The solve is
already pushing power down the gain spectrum.

Target projected onto the top-k span, Spearman over the 900-vertex edge sample: rank 10
+0.172, 20 +0.318, 47 +0.616, 100 +0.818, 200 +0.914, 400 +0.962, 600 +0.976. **The model
sits at +0.679 in that estimator with its variance spread over ~200 directions**, so it is
not at its span ceiling. And that ceiling rises **monotonically with edge length at every
rank** — it has no minimum in the middle, so the 10–30 mm dip is not a rank or conditioning
artefact.

### Cancellation does not set the spatial scale

Dialling the fitted input's inter-channel coherence, `S(λ) = λS + (1−λ)diag(S)` — λ=1 the
fit, λ=0 the same per-channel powers with every cross term removed:

| λ | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 |
|---|---|---|---|---|---|---|---|
| coherent/incoherent | 1.000 | 0.887 | 0.774 | 0.661 | 0.548 | 0.435 | 0.322 |
| FC correlation length (mm) | 21.8 | 21.9 | 22.0 | 22.1 | 22.3 | 22.6 | — |
| fit | +0.175 | +0.442 | +0.620 | +0.694 | +0.713 | +0.708 | +0.684 |
| variance in top-47 span | 0.499 | 0.492 | 0.484 | 0.472 | 0.455 | 0.427 | 0.400 |

Removing every cross term costs 0.54 of the fit and moves the correlation length by
**0.5 mm**. λ ≥ 1.25 is not PSD. So cancellation buys the fit, not the spatial scale, and
not the span occupancy either.

### The dip is in the covariance→correlation step

`solve` maximises `<C, Ct>/‖C‖` on the raw double-centred **covariance** with the diagonal
zeroed. `sim` standardises to a **correlation**. Same solved `S`, same `H`, same 900
vertices, pooled Spearman per band:

| band mm | 0–5 | 5–10 | 10–15 | 15–20 | 20–30 | 30–40 | 40–60 | all |
|---|---|---|---|---|---|---|---|---|
| C as covariance | +0.253 | +0.523 | **+0.579** | **+0.517** | +0.538 | +0.624 | +0.702 | +0.713 |
| C as correlation | +0.322 | +0.533 | **+0.380** | **+0.302** | +0.386 | +0.549 | +0.684 | +0.699 |
| cost of normalising | +0.069 | +0.009 | −0.199 | −0.214 | −0.153 | −0.075 | −0.017 | −0.014 |
| rank(σᵢσⱼ, C) | +0.977 | +0.943 | +0.782 | +0.568 | +0.245 | +0.095 | +0.067 | — |
| sd log σᵢσⱼ | 0.418 | 0.428 | 0.421 | 0.414 | 0.396 | 0.385 | 0.360 | — |

An iid draw from `C` at n = 100,000 reproduces the correlation row to three decimals
(+0.318 / +0.532 / +0.379 / +0.302), and the realised 14,263-frame run gives +0.343 /
+0.512 / +0.368 / +0.306 — so this is **not sampling**. Realisation length is worth about
+0.02 overall (577 s +0.638, 2,301 s +0.679, analytic +0.699).

**What picks the band is not the variance map's scale.** The nuisance injected is flat
across bands (sd log σᵢσⱼ 0.42 → 0.31), and the log-variance map's own autocorrelation
falls +0.899 → +0.383 over 0–60 mm, crossing half near 45 mm — four times the dip. What
varies is how much of the covariance edge the variance product already explains:
`rank(σᵢσⱼ, C)` runs +0.977 → +0.067, essentially the FC decay curve. Where it is near 1
dividing removes almost everything and leaves clean correlation shape; where near 0 it
changes nothing; the damage is maximal in between, on the shoulder of the model's own
22 mm correlation decay. Partial normalisation `C/(σᵢσⱼ)^α` at 10–15 mm: +0.579, +0.584,
+0.576, +0.533, +0.380 for α = 0, 0.25, 0.5, 0.75, 1.0.

Mean FC value by band, model correlation against target: +0.977/+0.809, +0.895/+0.548,
+0.770/+0.376, +0.626/+0.269, +0.429/+0.169, +0.257/+0.089, +0.145/+0.037. **The model is
about twice the target at every distance past 5 mm, and saturated near 1 below 10 mm.**

### The honeycomb does not reach the field

The frozen drive texture (0.39 on a third of the driven area, 0.86 on another third, at a
14.5 mm period) does not print through to the field's variance map:

| driven vertices | n | mean drive | mean variance | var / interior |
|---|---|---|---|---|
| internal seam | 515 | 0.401 | 1.98e-07 | **0.912** |
| outer edge | 493 | 0.410 | 1.77e-07 | 0.815 |
| piece interior | 524 | 0.868 | 2.17e-07 | 1.000 |

Drive contrast seam/interior is 0.461; the linear-response expectation for variance is
0.213; measured 0.912. Spearman(total drive, variance) inside the driven region **−0.001**.
Consistent with `interference.py`'s ~11 pieces per vertex: variance is dominated by the
mixture arriving through the medium, not by the local taper value. Conditions: reach 37 mm
against a 14.5 mm median piece, ratio 2.6.

Band accuracy is not organised by seam distance either. Spearman(acc_band, d_seam) −0.013;
partialling `d_drive`, −0.154, i.e. slightly worse *further* from a seam. Within the driven
region (`d_drive` 0–0.5) accuracy across 0–8 mm from a seam is +0.510 / +0.508 / +0.518 —
flat. The `d_drive` gradient §4 records does reproduce: +0.51 → +0.50s → +0.55s → +0.58 →
+0.588 down the strata.

### The incumbent solver stops 0.029 below its own maximum (`xspec.solve_factor`)

`solve` walks the PSD cone by projected gradient and stops on `--iters` — no gap, no
gradient test. `solve_factor` writes `S_f = L_f L_fᴴ` and runs L-BFGS on `L`: no PSD
projection (the constraint holds by construction), and the gradient is
`2·Hs_fᴴ(M B_f)` where `B_f = Hs_f L_f` is already built for the objective, so it costs one
extra gemm of the size the objective already pays.

On identical `H` (135, 1000, 47), `w`, `Ct`:

| | ρ | pearson vs raw | spearman vs raw | evaluations |
|---|---|---|---|---|
| `solve`, 400 PG steps | +0.732575 | +0.7564 | +0.7236 | — (83 s) |
| `solve_factor`, rank 8 | **+0.761495** | **+0.7843** | **+0.7552** | 9,465 (644 s) |

Terminating on `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`, not on budget.
`rank(Σ_f S)` from the projected gradient is **47/47** — full — where rank 8 suffices; ρ at
rank 16 matched rank 8 to 4e-05. Trace to convergence: +0.76027, +0.76093, +0.76116,
+0.76139, +0.761495.

**But sim fell.** `fac_r8` end to end: **+0.6577 ± 0.0008, gap 0.042, field rank 8.9**,
against the incumbent's +0.6901 ± 0.0018 / 0.065 / 20.5. In-sample objective up 0.029,
realised score down 0.032.

**That comparison is confounded and should not be read as "convergence hurts".**
`--rank r` is a hard structural constraint — the optimisation variable is `L` of shape
`(nf, K, r)`, so `rank(S_f) ≤ r` by construction, with no parameter able to express more.
`fac_r8` therefore changed two things at once: the search converged *and* the input
cross-spectrum was capped at 8 patterns per frequency where the incumbent ran uncapped at
47. The realised field rank of 8.9 is below the empirical 13.0. **The run that isolates
convergence is `--solver factor --rank 47`, and it has not been done.**

Two invariances survive the factorisation and are why it terminates on the function
tolerance rather than a vanishing gradient: ρ is scale-invariant, so ‖L‖ drifts freely —
it reached 7,225 by nfev 8,000, and a falling `|grad|` may be nothing but that, so read
`|grad|·‖L‖`; and `S` is unchanged by `L → LU` for unitary `U`, giving r² zero-gradient
directions per frequency.

### What this leaves

`--iters 400` is a regularisation parameter with a measured size: the incumbent sits 0.029
below the maximum of its own objective. Every configuration comparison in this file was
scored at that same fixed step count, including the spread/decay grid and the shell runs
above, so each carries some unmeasured amount of optimiser.

---

## 0++. 2026-09-08 (later) — the maps, the affinity target, affinity scoring

### The speed/damping maps are the largest lever measured, not a null

The brief carried "the speed/damping maps currently add little and it isn't known whether
that is a scale mismatch or COEF_LIM clipping". Neither holds.

**Not clipped.** `decode_maps` maps [0,1] linearly onto [-COEF_LIM, +COEF_LIM] and clips
nothing; the only clipping in the codebase is in `regime_deltas`, which is the
multi-regime path. The incumbent sits at `a = [-0.30, -0.05, 0.01]` and
`b = [-0.03, 0.35, 0.35]` against `COEF_LIM = 0.45` — 67% and 78% of the box, strictly
inside.

**Not weak.** The realised medium varies by **2.34x (p5-p95) in speed and 4.87x in
damping**, 5.74x and **18.1x** max/min across cortex. The box would allow 873x.

`--maps-scale` puts one knob through all six coefficients (0 = no cortical grading,
1 = incumbent, 1.28 = largest coefficient on COEF_LIM):

| `--maps-scale` | sim | gap | field rank | solve pearson |
|---|---|---|---|---|
| 0 — flat medium | +0.6606 ± 0.0003 | 0.123 | 12.0 | +0.7081 |
| **1.0 — incumbent** | **+0.6901 ± 0.0018** | 0.065 | 20.5 | +0.7564 |
| 1.28 — on COEF_LIM | +0.6665 ± 0.0038 | 0.063 | 21.0 | +0.7422 |

An interior maximum in sim and in the solve objective alike, sitting on the incumbent. So
the grading is worth **+0.030 of sim** — thirty times what the shell basis moved, and
comparable to the span of the whole spread × decay grid — and its overall STRENGTH is
already optimal under the current objective despite the coefficients being the `bo_step`
winner from before the clock retune, the RBC target and the passband.

The flat medium's field rank of 12.0 is the closest to the empirical 13.0 of any run while
scoring worst, another instance of rank and sim pulling apart. Its Moran gap of 0.123 says
the grading is much of what holds the model's spatial autocorrelation in range.

**What the knob cannot see is DIRECTION.** It scales all six together, so the mixture over
myelin / thickness / sulc is still whatever the old search chose, and no alternative map
basis has been tried. That is the open axis, and it now has a measured prize: a better
grading competes against +0.030, not against nothing.

### Fitting a low-rank affinity, scored against the raw FC (`affinity_lowrank`)

`gradients.affinity_lowrank` builds the Margulies affinity — top (1-thresh) of each row,
zero the rest and negatives, row-normalise, cosine — and reconstructs from the leading K
eigenpairs. Only the solve sub-block is formed, via a LinearOperator on `W W^T`.

Why the affinity rather than a plain PCA of FC: attenuation is multiplicative,
`R_obs = D R D` with `a_i = sqrt(SNR/(1+SNR))`, and a low-rank truncation leaves `D`
untouched since `D R D` has the same rank as `R`. The cosine removes it — row i is
`a_i (R D)_i`, so `a_i` and `a_j` cancel exactly. The ROW-PERCENTILE threshold is what
buys that; an absolute threshold would put the scale back.

`--target-affinity K` changes only what the solve is handed, so scoring stayed on the raw
FC and +0.6901 remained the comparator:

| fit target | ceiling (Â vs raw) | solve pearson | sim | gap | rank | sim / ceiling |
|---|---|---|---|---|---|---|
| **raw FC — incumbent** | 1.000 | +0.7564 | **+0.6901 ± 0.0018** | 0.065 | 20.5 | 0.690 |
| affinity K = 3 | +0.6472 | +0.5841 | +0.5587 ± 0.0039 | 0.108 | 8.5 | 0.863 |
| affinity K = 10 | +0.8263 | +0.7295 | +0.6398 ± 0.0087 | 0.081 | 17.9 | 0.774 |
| affinity K = 30 | +0.8551 | +0.7358 | +0.6418 ± 0.0023 | 0.077 | 19.9 | 0.751 |
| affinity K = 90 | +0.8657 | +0.7338 | +0.6490 ± 0.0060 | 0.086 | 18.7 | 0.750 |

The Â-vs-raw Spearman is a **ceiling**: a perfect fit to Â scores exactly there against
raw. At K = 3 that ceiling (+0.6472) is already below the incumbent's achieved +0.7236, so
that cell could not help whatever the model did.

The decomposition is exact. The affinity IS the easier target — the model reaches **75% of
its ceiling against 69% for the raw FC**, stable across K = 10-90 — but it asymptotes at
0.866 agreement with the raw FC, and 0.75 x 0.866 = 0.649, the observed sim, against
0.69 x 1.0 = 0.690. Efficiency up 9%, ceiling down 13%.

**The protocol is what limits this, not the idea.** Fit-to-Â/score-to-raw structurally
disadvantages any denoising: the denoised target is by construction further from raw, so
the ceiling always falls and the method must overcome that from efficiency alone. It is
the right test of "does this improve the number we report" and not of "does this get closer
to the underlying field", and those come apart exactly when the target is noisy — which was
the reason for trying it.

### Scoring in affinity space (`--score-affinity`)

`FCTarget.attach_affinity` / `.affinity_sim` score a realisation through the same recipe on
both sides, reusing the same `self.i/self.j` edge sample, the same double-centring and the
same rank-and-normalise `_prep` — the same estimator as `sim`, in a different space. The
transform sits OUTSIDE the solve, so the objective stays linear in S and convex.
`score_realisation` reports `sim_aff` whenever it is attached, so pooled draws get it too.

Both sides use the affinity of the DOUBLE-CENTRED FC, which departs from the Margulies
convention of the raw correlation matrix but keeps model and target identical.

| fit target | **AFFINITY sim** | raw sim | gap | rank |
|---|---|---|---|---|
| raw FC — incumbent, 2 draws | +0.7326 ± 0.0108 | +0.6901 ± 0.0018 | 0.065 | 20.5 |
| affinity K = 30, 2 draws | +0.7546 ± 0.0112 | +0.6418 ± 0.0023 | 0.077 | 19.9 |
| affinity K = 90, 2 draws | +0.7585 ± 0.0063 | +0.6490 ± 0.0060 | 0.086 | 18.7 |
| **raw FC — REFERENCE, 4 draws** | **+0.7300 ± 0.0083** | **+0.6915 ± 0.0026** | 0.073 | 19.7 |
| affinity K = 90, 4 draws | +0.7489 ± 0.0108 | +0.6462 ± 0.0051 | 0.086 | 18.6 |

**The printed ± is `np.std` across draws, not the standard error.** Earlier readings that
used it directly as an SE understated the significance. With SEM = sd/sqrt(n), fitting the
affinity leads by +0.0259 ± 0.0088 at 2 draws (2.9 sigma) and +0.0189 ± 0.0068 at 4
(2.8 sigma) — the point estimate falls, the significance holds.

**Adopted: fit AND score the K=90 affinity** (`--target-affinity 90 --score-affinity`).
The fit target costs **2.0 s**, once per run — `affinity_lowrank` takes the top-90
eigenpairs through a LinearOperator on `W W^T` and never forms it — so the compute
objection to it does not apply. The remaining trade is +0.019 in affinity space against
-0.045 in FC space, taken deliberately: the gradient decomposition is the currency the work
will be reported in.

**Reference for everything after this: `as_k90_4`** — AFFINITY sim **+0.7489 ± 0.0108**,
raw sim +0.6462 ± 0.0051, gap 0.086, field rank 18.6, at 4 draws.

Affinity costs, measured: `target_fc` 0.2 s, `affinity_rows` 0.5 s, `affinity_lowrank`
K=90 **2.0 s once per run**, `affinity_edges` on 2M edges **16.0 s once per DRAW**. The
16 s is the price of `--score-affinity` whichever target is fitted.

Read with two cautions. Part of the gain is guaranteed rather than earned, since the
objective and the metric are now aligned — though not all of it, because the fit is against
a rank-K affinity while the score uses the full-rank one. And both affinity-fit arms are
WORSE in FC space (+0.6418, +0.6490 against +0.6901), so this is a trade between metrics,
not an improvement in both.

**The affinity metric is noisier**: draw scatter ±0.006-0.011 against ±0.002-0.006 for the
raw FC, so at two draws a difference needs roughly ±0.015 to mean anything and these sit
just above it. AFFINITY numbers are NOT comparable to raw-FC sim; both are reported so the
older record stays readable.


### Two fancy-index chunks were holding multi-GB transients

Affinity scoring needs full FC ROWS, so unlike the edge-sampled `sim` it has to form the
model FC. Put into `score_realisation` — the function the draw pool calls — at the pool's
default width, seven workers each opened a full-width BLAS matmul at once and the host went
down.

The size was not where it looked. `model_edges` and `affinity_edges` both index the edge
endpoints in chunks of a FIXED 250,000, and `W[i[b]]` is a contiguous COPY of shape
(chunk, row width). At 9,310 columns that is 9.3 GB per operand; in `model_edges` at a
2,308 s realisation, with T = 14,263, it is **14.3 GB** — and `model_edges` is on every
scoring path in the repo, so that transient long predates any of this. Both chunks are now
sized from the row width against a 512 MB per-operand budget. Values are identical; only
the loop count changes.

Measured at 577 s, 3 draws:

| | no affinity | with affinity |
|---|---|---|
| fixed 250,000 chunk | 20.07 GB / 51.8 s | 31.93 GB / 96.9 s |
| 128 MB budget | 13.90 GB / 71.7 s | 15.37 GB / 89.8 s |
| **512 MB budget** | **14.62 GB / 52.3 s** | **16.12 GB / 85.9 s** |

512 MB keeps the original speed and 27% of the memory saving on the core path; 128 MB cost
38% wall because chunks of ~8,900 give the einsum too little to work with. At 2,308 s the
pool workers went from 8.7 GB each to 0.3-0.4 GB.

`parallel_scores` also caps the pool when affinity scoring is attached. The first cap —
2 workers x 2 threads — was an over-correction: the thread limit throttles `fl.run`, which
is the bulk of a draw and has nothing to do with the affinity, turning a 25-minute arm into
70+ minutes at 5-way load. It is now 4 workers x 4 threads.


### A reduced solve ranks media correctly; a CONVERGED one does not

The three `--maps-scale` media have known full-pipeline scores, so they are a ready-made
test of whether a cheap surrogate preserves the ordering a medium search would rely on.
Ordering to reproduce: **1 > 1.28 > 0** (+0.6901, +0.6665, +0.6606), the interior maximum
being the feature that distinguishes "stronger grading is better" from "the incumbent is at
an optimum". Impulse responses were cached for all three, so this isolates solve cost.

| budget | s=0 | s=1 | s=1.28 | order | wall/medium |
|---|---|---|---|---|---|
| nvert 1000, 400 iters | +0.6875 | +0.7326 | +0.7159 | 1 > 1.28 > 0 | 85.1 s |
| nvert 1000, 100 iters | +0.6524 | +0.7055 | +0.6866 | 1 > 1.28 > 0 | 22.5 s |
| nvert 400, 400 iters | +0.6791 | +0.7396 | +0.7314 | 1 > 1.28 > 0 | 43.2 s |
| nvert 400, 100 iters | +0.6386 | +0.7082 | +0.6969 | 1 > 1.28 > 0 | 11.2 s |
| **nvert 400, 50 iters** | +0.6138 | +0.6896 | +0.6741 | **1 > 1.28 > 0** | **5.3 s** |
| nvert 400, rank 8, 300 fev | +0.7760 | +0.7781 | +0.7616 | 1 > 0 > 1.28 | 8.6 s |
| nvert 400, rank 8, 1500 fev | +0.7962 | +0.7881 | +0.7701 | **0 > 1 > 1.28** | 40.8 s |

Every projected-gradient budget holds the order down to **5.3 s per medium, 16x faster**
than the 1000/400 reference. It preserves rank and NOT spacing — at 400/50 the 1-vs-0 gap
reads 0.076 against the reference's 0.045 — so it ranks media and cannot be read for effect
sizes.

**The factorised solve inverts the comparison.** Converged at rank 8 the FLAT medium scores
highest (+0.7962) where the full pipeline puts it last (+0.6606 against +0.6901). Given
enough input freedom and an exactly-maximised objective, an ungraded medium fits the
covariance pattern as well as a graded one: the medium's contribution is absorbed into `S`.

So `fac_r8`'s "higher objective, lower sim" is not a shifted number, it REVERSES model
comparisons. The incumbent's under-convergence at 400 steps is load-bearing for model
selection, and `--iters` is better read as what makes the objective discriminate between
media than as a defect to be engineered away. Three points is a weak test — it says the
surrogate is not broken, not that it tracks the pipeline across medium space.


### The surrogate holds under the affinity target

Re-run with the K=90 affinity as the solve target, same three media:

| budget | s=0 | s=1 | s=1.28 | order | wall/medium |
|---|---|---|---|---|---|
| nvert 1000, 400 iters | +0.6943 | +0.7264 | +0.7038 | 1 > 1.28 > 0 | 91.3 s |
| nvert 400, 100 iters | +0.6276 | +0.6865 | +0.6685 | 1 > 1.28 > 0 | 11.4 s |
| **nvert 400, 50 iters** | +0.6051 | +0.6656 | +0.6447 | **1 > 1.28 > 0** | **5.5 s** |
| nvert 400, 25 iters | +0.5790 | +0.6423 | +0.6184 | 1 > 1.28 > 0 | 3.1 s |

Still unverified: **1 > 1.28 > 0** is the FC-space ordering; those three media have no
full-pipeline scores under the adopted affinity configuration.

### A medium search costs 31 s a candidate, and 82% of it is the impulses

`xspec.impulse_responses(..., cache=False)` was added because the cache key carries `a` and
`b`, so every candidate medium is a miss and each 47-piece set is **1.79 GB** against
101 GB free — a search would fill the disk after ~56 candidates. Cold, uncached, 47 pieces
over 8 workers: **25.1 s**, plus 5.5 s for the reduced solve. Further speedup belongs on
the impulse stage, not the solve.

### LB eigenmodes as a map basis (`cortical_maps.lb_modes`)

Built from the SAME discrete operator the medium integrates — swe_rot's `grad` is
`(h[Ej]-h[Ei])/d` and `divergence` accumulates `±l·ue` then divides by `A`, so
`div(grad h)_i = (1/A_i) Σ_j w_ij (h_j - h_i)` with `w = l/d`. The modes are the
generalised eigenvectors of `K φ = λ A φ`. `load_maps` resolves names `lb<k>`, so a
map-graded medium takes a geometric basis with no change downstream; `fluid.map_fields`
only asks for names and stacks what it gets. This grades a medium PROPERTY while the drive
stays localised and anatomical — the opposite of driving eigenmodes directly, which returns
a global pattern for a global input.

**The basis is smoother than the anatomical maps, not more expressive.** Fraction of each
map's variance inside the first J modes:

| J | myelin | thickness | sulc | min wavelength |
|---|---|---|---|---|
| 4 | 0.368 | 0.423 | 0.014 | |
| 16 | 0.591 | 0.607 | 0.077 | |
| 32 | 0.678 | 0.671 | **0.185** | 75.5 mm |
| 128 | 0.857 | 0.787 | 0.535 | |
| 256 | 0.913 | 0.859 | **0.776** | 26.6 mm |

Myelin and thickness are largely smooth; **sulc is not** — it varies at the gyral scale and
needs hundreds of modes. The incumbent's damping is `b = [-0.03, 0.35, 0.35]` on
myelin/thickness/**sulc**, so its two large coefficients sit on the map the eigenmode basis
represents worst. At any tractable J the LB basis is a strictly smoother family that does
NOT contain the current best medium, which bounds smooth grading rather than grading.

---

## 0+++. 2026-09-09 — searching the medium's grading (`fit/map_search.py`)

`map_search.py` searches the speed/damping map coefficients with the reduced solve
(nvert 400, 50 iterations, K=90 affinity target) as the objective, seeded at the incumbent
so generation 0 contains the current best medium exactly. `--check-every` re-evaluates the
best at a STRICT budget (nvert 1000, 400 iterations) and logs it beside the surrogate's
claim, because the first search's gain did not survive.

### The surrogate can be optimised against, and overstates by ~7x

Anatomical + `lb1..lb8`, 22 coefficients, 169 candidates: surrogate **+0.069015**. Scored
through the full pipeline (2,308 s, 4 draws, K=90 affinity fit and score) as `ms_bp`:

| | AFFINITY sim | raw sim | gap | field rank |
|---|---|---|---|---|
| incumbent `as_k90_4` | +0.7489 ± 0.0108 | +0.6462 ± 0.0051 | 0.086 | 18.6 |
| searched `ms_bp` | +0.7583 ± 0.0164 | +0.6340 ± 0.0118 | 0.070 | 27.4 |

**+0.0094, and with SEM = sd/sqrt(4) the standard error on the difference is 0.0098 —
0.96 sigma.** The surrogate promised +0.069 and delivered a seventh of it, indistinguishable
from zero. The reduced solve ranks the three known media correctly (§0++) but that
established it was not broken, NOT that it could be optimised against with 22 free
parameters. Same family as the converged-solve inversion in §0++.

The strict check exists to make that visible while a search runs. Measured ratios of
strict to surrogate gain: 36-45% across capped and uncapped runs, against the 14% the
pipeline delivered.

### What the searches keep finding: myelin in both roles, opposite signs

The incumbent uses ONE MAP PER ROLE — myelin sets speed (a=-0.300, b=-0.030), thickness and
sulc set damping (b=+0.350 each, a within noise of zero). No map carries a substantial
coefficient in both. Across cortex that gives **corr(log c, log sigma) = +0.332**, mildly
POSITIVE.

Every search so far moves off that arrangement the same way: damping onto myelin, sulc out
of damping, and the speed/damping correlation flipped negative.

| medium | corr(log c, log s) | speed | damping | reach |
|---|---|---|---|---|
| flat (`--maps-scale 0`) | — | 1.0x | 1.0x | 1.0x |
| **incumbent** | **+0.332** | 5.7x | 18.1x | 19.0x |
| anatomical, uncapped (63 cand) | -0.349 | 14.7x | 18.2x | 51.6x |
| LB, uncapped (169 cand) | -0.425 | 12.1x | 39.1x | 230.0x |

Anti-correlating speed and damping is the efficient way to manufacture reach heterogeneity,
and the searches reach for it whatever basis they are given: the largest single `a-b` was
`lb2` at -0.606 in the eigenmode run and **myelin at -0.629** in the anatomical one. With
myelin's speed coefficient negative, a positive damping coefficient means high myelin =
slower AND more damped = short reach; low myelin = fast, lightly damped, long reach. That is
the unimodal-to-transmodal axis, and the same axis the affinity target is built from.

### The flat control, and where the power actually is

`--maps-scale 0` gives a uniform medium. All three at 2,308 s / 4 draws, K=90 affinity:

| medium | AFFINITY sim | raw sim | gap | rank | raw power in 0.01-0.08 Hz | filter amp |
|---|---|---|---|---|---|---|
| flat | +0.7119 ± 0.0079 | +0.6200 ± 0.0042 | 0.135 | **10.9** | **9.8%** | 0.24x |
| **incumbent** | +0.7489 ± 0.0108 | +0.6462 ± 0.0051 | 0.086 | 18.6 | 17.2% | 0.38x |
| searched (LB) | +0.7583 ± 0.0164 | +0.6340 ± 0.0118 | 0.070 | 27.4 | 22.1% | 0.48x |

- **flat -> incumbent: +0.0370 ± 0.0067, 5.5 sigma.** The grading does real work.
- **incumbent -> searched: +0.0094 ± 0.0098, 0.96 sigma.** Twelve times the reach spread
  buys nothing measurable.
- In-band power rises MONOTONICALLY with grading strength, 9.8% -> 17.2% -> 22.1% across a
  230-fold range of reach spread. Grading pulls the medium's temporal content INTO the
  scored band rather than pushing it out — the opposite of the reading I first gave.
- **Real BOLD carries 84% of its variance in 0.01-0.1 Hz (§1).** The most heavily graded
  medium here reaches 22.1%. A 23-fold increase in reach heterogeneity bought 7 percentage
  points and BOLD is 62 points further on. The multi-timescale mismatch is large and the
  grading does not close it.

`viz/render_bandpass.py --tag <tag> --resimulate` renders the pair; `resimulate_raw` now
prefers the run's OWN saved map basis, since `x` encodes only three anatomical coefficients
and a richer medium would otherwise be silently resimulated as the incumbent — same drive,
wrong physics, and the two rows would differ in more than the filter. Videos:
`bandpass_pair_{flat_bp,as_k90_4,ms_bp}_200-800.mp4`.

### Capped and staged (`--reach-cap`, `--damping-cap`, `--stage`, `--init`)

Caps bound the REALISED spread, computable without simulating. Enforced by scaling the free
block by the largest t in [0,1] that satisfies both, by bisection on a linear form — not by
rejection, so no evaluation is wasted, and `best_x` is stored already projected. An earlier
version scaled `(a-b)`, which rewrites `a` and dragged speed along during the damping stage,
defeating the staging; scaling only the free block fixes it. t=0 is always feasible.

**Both caps are needed.** Reach is c/sigma, purely spatial: a medium can hold it constant
while scaling c and sigma together, giving identical spatial structure and a wildly varying
TIMESCALE. Since decay time is 1/sigma, the damping cap is what bounds temporal scales.

Caps set at the incumbent's own values — reach 19.0x (it runs at 18.967x), damping 18.1x
(18.064x) — so the search may rearrange its heterogeneity but not increase it.

**Stage 1, damping (speed held), 98 candidates:**

| map | a (held) | b incumbent | b stage-1 |
|---|---|---|---|
| myelin | -0.300 | -0.030 | **+0.158** |
| thickness | -0.050 | +0.350 | +0.125 |
| sulc | +0.010 | +0.350 | **+0.004** |

surrogate +0.682439 (**+0.0168**), strict +0.733306 (**+0.0069**). Its damping spread is
**3.458x against the incumbent's 18.064x** — constrained to the incumbent's limits, the
search found a medium with a FIFTH of the temporal heterogeneity that scores slightly
better, where the unconstrained runs had gone the other way to 39x.

**Stage 2, speed (damping held), in progress.** At generation 5: surrogate +0.732154,
strict +0.751920. Cumulative over the ORIGINAL incumbent: surrogate **+0.0665**, strict
**+0.0255** — matching the unconstrained LB run's +0.069 surrogate while holding reach at
19x and damping at 18.1x. Note `--init` reseeds the baseline, so stage 2's printed gains are
against stage 1, not the original.

**Neither staged medium has been scored through the pipeline yet.** Given the LB run
delivered 14% of its surrogate claim, that re-score is what decides whether any of this
holds.

### Maps are z-scored, and right-skewed

`load_maps` returns z-scored maps (means 2e-09, 0, -1e-09; sd 1.000), so c0 and sigma0 are
the GEOMETRIC means of the realised fields — both ratios come out at 1.0000 — and there are
always vertices on both sides. Not symmetrically: all three maps are right-skewed (myelin
-2.52 to +3.97), so in the incumbent 58.1% of vertices are faster than baseline and 41.9%
slower, and the arithmetic means sit above it (1.035 speed, 1.111 damping) since exp is
convex. A coefficient buys a long fast tail and a short slow one. The z-scoring is also
unweighted, so by area the maps are slightly off-centre (-0.076 to +0.045).

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

### The sweep at 577 s, 40 runs (`bp_sweep.py`)

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
- ~~**sim is flat in decay above ~25 s**~~ — 1.47 ran +0.597 to +0.633 across a 4× range
  non-monotonically, with per-run scatter up to 0.016. **That plateau was sampling noise.**
  At 2,308 s the same row rises monotonically to decay 25 with non-overlapping bars; see
  below.
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

### Realisation length was never required to be 577 s

577 s matches one NKI run, but the target is a 100-subject Fisher-z average — close to the
population covariance — while the model side was a single 577 s draw carrying its own
estimation noise. Nothing makes the two share a duration. Realising longer from the SAME
saved `S` (no re-solve, so the fit is identical and only the sampling changes):

| frames | seconds | sim | rank | wall |
|---|---|---|---|---|
| 3,578 | 577 | +0.6488 | 15.2 | 24 s |
| 7,156 | 1,154 | +0.6605 | 18.2 | 63 s |
| 14,312 | 2,308 | +0.6899 | 19.7 | 229 s |
| 28,624 | 4,616 | **+0.7004** | 20.9 | 1,288 s |

+0.052 for the identical fit, more than the whole spread × decay grid gained over its own
starting point. A bandpassed model sampled for 4,616 s scores about what an unfiltered one
sampled for 577 s does (+0.7107). Rank moves with it too, 15.2 → 20.9, so part of what
reads as the model overshooting the empirical 13.0 is duration, not medium — and the
empirical 13.0 is itself measured on ~900-frame runs. Cost scales worse than linearly.

Distinct from `--pad`, which sets the transfer function's frequency resolution: that was
tested at 16384 and landed inside the draw scatter.

### The impulse window was too short for the passband

The auto-bump set the window to exactly 3 decay times, cutting the response while still at
exp(−3) = 5% of peak. Measured on the decay-25 medium against a 13-decay-time reference:

| window | decay times | tail/peak | mean \|ΔH\|/\|H\| in 0.01–0.08 Hz |
|---|---|---|---|
| 512 (83 s) — what the 577 s grid used | 3.3 | 2.70e-02 | **4.63%** |
| 1024 (165 s) | 6.6 | 6.63e-03 | 0.18% |
| 2048 (330 s) | 13.2 | 4.94e-04 | reference |

H is what the convex solve optimises against, so every bandpassed fit above carried that
4.6%. Under a broadband objective it was spread over all frequencies; the passband IS the
low-frequency end. The shortest cells were worse: spread 1.5 / decay 12 ran a **36 s**
window against a **100 s** cycle at 0.01 Hz, because 224 frames just cleared 3 × 74 and no
bump fired — so the bias ran along the decay axis being swept, not as a constant offset.
`--impulse-decays` sets the multiplier, still defaulting to 3 so recorded runs reproduce;
with `--bandpass` the window must also hold one cycle of the lowest passband frequency.

### The grid re-run at 2,308 s with 7-decay windows (`g7_*`)

|spread|d12|d15|d20|d25|
|---|---|---|---|---|
|1.47|+0.6357|+0.6546|+0.6771|**+0.6901 ± 0.0018**|
|1.84|+0.6482|+0.6584|+0.6797|+0.6836|
|2.58|+0.6406|+0.6485|+0.6439|+0.6508|
|2.95|+0.6262|+0.6342|+0.6331|+0.6289|

**+0.6901 ± 0.0018 at spread 1.47 mm/s, decay 25 s** — the same cell as at 577 s, so
neither change moved the optimum. Against +0.4701 for the old 5.89 / 9.03 medium under the
same filter, and +0.7102 unfiltered.

- **The window fix is small and one-directional.** Against the four cells run at 2,308 s
  with short windows: +0.6325→+0.6357, +0.6489→+0.6546, +0.6708→+0.6771, +0.6861→+0.6901.
  Every one gains, +0.003 to +0.006 — about the draw scatter, and far less than length.
- **Draw scatter collapses** to 0.0001–0.0072 from up to 0.0231. The whole 1.47 row is now
  ordered with non-overlapping bars where decay 25/50/100 had been indistinguishable.
- **The rows compress.** 2.95 is 0.06 below 1.47 at decay 25 but only 0.010 at decay 12, so
  sim resolves spread less sharply than at 577 s. Moran gap still separates them cleanly
  and monotonically — 0.065–0.076 at spread 1.47 against 0.131–0.133 at 2.95 — as does
  rank, 17.4–20.5 against 11.2–12.2, with the empirical 13.0 nearest the WORST row.

### Edge correlations

`sim` is Spearman: `_prep` ranks both edge vectors, centres and normalises, so the score is
a normalised dot product of ranks over the fixed 1,999,793-edge sample, both matrices
double-centred (`metric=spearman, centre=double`). Pearson runs about 0.037 above it:

| run | edge pearson | edge spearman | model edge sd |
|---|---|---|---|
| best bandpassed | +0.6863 | +0.6488 | 0.2311 |
| unfiltered | +0.7599 | +0.7107 | 0.1801 |

Target edge sd is 0.1302, so the model's edge distribution is **1.8× too wide** under the
bandpass against 1.4× without it. Whether the model's FC is built by ranking the
timecourses first barely matters — +0.6863 against +0.6852 for the same run.

### The fitted input is self-cancelling (`interference.py`)

The medium is linear, so the field is a superposition of one contribution per piece and
each can be formed separately from the cached impulse responses — no new simulation. Per
vertex, `coherent = var_t(Σ_k f_k)` against `incoherent = Σ_k var_t(f_k)`: what the field
does, against what it would do if the pieces never met.

Summed over cortex the ratio is **0.552**, and **96.0% of vertices are below 1**. The
pieces produce a little over half the variance they would produce independently. Frame by
frame the cortex-summed cross term is **never positive** in 600 frames (mean −0.462, range
−0.639 to −0.095), so this is a standing arrangement rather than a beat; locally 79.5% of
vertex-frames cancel and about a fifth reinforce. Each vertex is fed by ~11 pieces
(participation ratio of the variance share, median 10.8, p5–p95 3.7–21.5), and that is
nearly independent of how much they cancel (r = +0.149).

**No contributor resembles the result.** Each piece's contribution against the total, over
vertices × time: max +0.236, median +0.107; `Σ r² = 0.636` where orthogonal pieces give
1.0. Each piece's OWN FC against the target: max +0.278, median +0.176 — against the
combination's +0.733. The fit is emergent, not carried by any source.

Verified before use: the superposition reproduces the run's saved frames at r = 0.998965
(sd ratio 1.008, the residual being the per-step drive against per-`save` responses), and
the analytic route — `coherent` and `incoherent` straight from S and H — agrees with the
time-domain one at 0.545 vs 0.552, per-vertex r 0.997/0.999. The analytic H must carry the
BANDPASS as well as the smoothing kernel; with only the kernel it reads 0.837.

### How much of that the fit requires (`xspec.incoherent_reg`, `family_member`)

Incoherent power is linear in `diag(S)` with fixed weights, so it drops into the existing
feasible-direction machinery: minimise it with the fit held within `eps` of the argmax,
trace preserved. Not a penalty weight — `family_member`'s docstring gives the reason.

| eps | fit | incoherent | coherent/incoherent | stopped on |
|---|---|---|---|---|
| 0 | +0.7564 | — | 0.544 | — |
| 0.0002 | +0.7562 | −2.7% | 0.552 | fit |
| 0.001 | +0.7554 | −10.2% | 0.578 | fit |
| 0.01 | +0.7464 | −36.0% | 0.676 | fit |
| 0.05 | +0.7064 | −62.6% | 0.783 | fit |
| 0.1 | +0.6564 | −74.8% | 0.852 | fit |
| 0.2 | +0.5564 | −84.9% | 0.942 | fit |
| **0.35** | **+0.4064** | −90.3% | **0.999** | fit |
| 0.5 | +0.3183 | −92.0% | 1.000 | **R** |

**There is no slack**: every eps down to 0.0002 ends with the fit constraint binding, so
cancellation cannot be reduced at zero cost. What is true is that the exchange rate is
steep — 0.1% of fit buys a 10% reduction — and smooth over two and a half decades with no
knee. The objective is a scale-invariant correlation of PATTERNS and never sees how much
opposing power bought them, so the amount present at the argmax is only weakly determined.

**Zero cancellation costs 46% of the fit.** A fully non-cancelling input exists and scores
+0.3183; beyond eps=0.35 the search stops on R rather than on the fit and lands in the same
place at every larger eps, so a ratio above 1 is unreachable — these 47 pieces can be made
not to interfere, never to reinforce.

These are lower bounds on how much cancellation could be shed: `family_member` finds *a*
member within eps, not provably the minimum, and stopped after ~30 accepted steps each time.

---

> **Sections 1–9 were all measured before 2026-09-04**, against the nilearn target, with
> no bandpass on the observable, at a 577 s realisation and a 3-decay-time impulse window.
> Section 0 changed all four. They remain comparable with each other and are not restated
> here; individual claims that section 0 actively CONTRADICTS are marked inline below.
> Claims merely measured under other conditions are left alone — that is what section 3's
> own preamble already says about itself.

## 1. The clock (`timescale.py`)

`c0` is not a speed knob and never was: `dt = CFL·d_min/c` means distance per step is
`CFL·d_min` whatever `c` is (verified identical at c0 = 0.25, 1, 4). `save` sets distance
per frame. Seconds per frame is a free anchor, and it had been set two incompatible ways:

- `units.py` declared a 300 mm/s spread → one frame = 6.54 ms
- `bo_step` matched model-frame FC to TR-sampled FC → one frame = one TR

They differ by 99x. Under the first, a 1,120-frame window is 7.3 s and its lowest
frequency is 0.137 Hz — above the entire resting-state band, so restricting the solver to
what BOLD sees was not even expressible.

**Both anchors are now measured from NKI:** — from *nilearn's* NKI. Re-measured on RBC's
XCP-D preprocessing they read 1/e 4.06 s and slope −0.90 with 94.7% of power in band, but
those describe the 0.01–0.08 Hz filter as much as the brain (§0).


| quantity | value |
|---|---|
| BOLD spectrum | `f^-2.60`; 84% of variance in 0.01–0.1 Hz, 66% in 0.01–0.03 |
| BOLD autocorrelation | 9.03 s at 1/e, 14.9 s integrated → 38.8 independent samples per 577 s |
| single-run field effective rank | 10.9–13.4 |
| group FC effective rank | 90.2 |

~~Decay is pinned to the data; spread is the one free parameter~~, with reach = spread ×
decay. At TR/4: frame 0.161 s, `save` 8, damping 2.23e-3, 3,578 frames = 577 s.
**Superseded (§0):** decay was pinned to a 1/e measured on UNFILTERED nilearn data and
applied to the medium's PRE-filter decay. Under the passband it is not recoverable from
the filtered data either — a 60× range of input decay maps onto 4.29–7.70 s out — so both
spread and decay are now swept against the score. Reach does not summarise the pair: at
matched reach, spread 2.95 × decay 9.03 gives +0.5730 and 1.84 × 15 gives +0.6192.

**Frequency resolution matters more than the medium did.** 0.01–0.03 Hz holds 63% of BOLD
power and a 165 s window resolves three bins there. Padding 1024 → 4096 frames (661 s,
0.0015 Hz bins) moved the score +0.291 → +0.504 with no model change. ~~Zero-padding is
free because the impulse has decayed by frame 224.~~ **Corrected (§0):** at frame 224 the
impulse is at 5% of peak, not zero, and that truncation is 4.6% error in H across
0.01–0.08 Hz. Padding further does not recover it — 16384 landed inside the draw scatter —
because zero-padding cannot restore a tail that was never integrated.

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

**Not re-measured since §0.** Everything here is the nilearn target, unfiltered, at 577 s,
and the whole section turns on a length scale — the band ceiling of +0.9119 is a split-half
of those 99 subjects, and the model's +0.5793 is a medium running at 5.89 mm/s. The
bandpassed optimum runs at 1.47 mm/s with an in-band FC correlation length of 22 mm
against 71 mm, which is the same scale this band sits at, so whether the dip survives is
open rather than answered. `diag_edges.py` and `band_fail.py` on a `g7_*` tag would settle
it.

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
   ceiling budget — this is the live question. **Worse under the passband (§0):** against
   the RBC target the ratio is 1.8× bandpassed against 1.4× unfiltered, so the filter
   amplifies exactly this.
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
   rank 25.6 against the target's 90.2. **Recheck before reuse (§0):** that 25.6 is an
   unfiltered 577 s figure. Bandpassed the same medium gives 4.1, and length alone moves
   rank by a third, so the 5% cap is not a fixed property of the model.

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
~~On the fMRI clock, quote 577 s (3,578 frames at TR/4) so it matches the data's
duration.~~ **Superseded (§0):** matching the data's duration was never required — the
target is a 100-subject average and only the model side carried the sampling noise.
Length is worth +0.052 from 577 s to 4,616 s on an unchanged fit, so it has to be quoted
with every score, and scores at different lengths are not comparable.

Field rank has a reference now: a single fMRI run measures **10.9–13.4**. The group FC's
90.2 is an average over 99 subjects and is NOT what one simulated run should match —
`fc_score.attach_rank`'s docstring cites 95 for "the empirical FC" and that is the wrong
comparison. **Two conditions on that reference (§0):** it is a nilearn number; on RBC's
bandpassed runs the same measure gives **13.0** (8.8–17.9 over 20 subjects). And rank
depends on how long the model was sampled — 15.2 at 577 s rising to 20.9 at 4,616 s for
one unchanged fit — so a rank comparison is only meaningful at matched duration, which the
empirical ~900-frame runs and a 577 s realisation roughly are.

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
