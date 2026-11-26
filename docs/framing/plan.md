# Plan: what we keep, the rerun, and how we present it

Started 2026-11-25. A working plan, edited as decisions are made. The argument it serves is in
`core_argument.md`. The driven-reservoir reading belongs to the introduction and discussion
only, not to methods or results.

## Decisions taken

- **The uniform medium is the reference.** Heterogeneity is a designed manipulation on the
  medium axis, not a fitted nuisance (2026-11-25).
- **The target is the unfiltered FC** (`results/fc/nobp/`, settled in October). The usual
  0.01–0.08 Hz bandpass is a convention for suppressing non-neural noise, not evidence that
  the signal lives there; what fMRI measures is roughly scale-invariant across the range
  it samples (2026-11-26).
- **The observable is the field low-passed only to what the TR can report**
  (`--bold-smooth`). No bandpass, no HRF and no added global-signal regression in the fit.
  The canonical HRF is only superficially sensible as a description of how the field
  becomes BOLD, which is why the project never fitted through it (2026-11-26).
- **Toy examples open the presentation.** They are built after the reruns, not from fits. They
  introduce an input-driven medium on a geometry, action at a distance, and input timing
  shaping waves like a phased array.

## Why the map weights are no longer justified

The map weights (`BEST_X` in `fit/best_fit.py`) came from Bayesian optimisation
(`fit/bo_step.py`), not from the genetic algorithm, which searched inputs and was removed on
2026-08-27. They were searched jointly with damping, rotation, boundary absorption and frame
cadence, on the old clock (~99x too fast), against the nilearn unfiltered target, with the
sensory input basis, 280 frames and a fixed-iteration solve, before every later bug fix.

| map | speed weight | damping weight |
|---|---|---|
| myelin | -0.30 | -0.03 |
| thickness | ~0 | +0.35 |
| sulcal depth | ~0 | +0.35 |

Two weights sit near the +-0.45 search limit. Since then, clipping the maps' tails cut the
depth range from 33x to 5.9x. A dedicated search for better weights gained nothing
measurable (0.96 sigma). In the configuration the wave analysis uses, the graded medium
scores *below* the uniform one (3x: 0.739 against 0.757).

Designed gradings should use maps whose meaning belongs to the medium (myelin, thickness,
cell or receptor density). Sulcal depth is a property of geometry, so grading the medium by
it mixes two axes.

## Phase A: decision experiments

These run before anything is frozen, because they can change the story.

| # | experiment | decides | status |
|---|---|---|---|
| A1 | FC score under the independent, random-coherent and timing-only controls, for the four wave-analysis media | whether FC requires the fitted lags that organise the waves | 3x **done** ([findings](phaseA_findings.md#a1-fc-score-under-the-controls)): every control drops FC from 0.757 to 0.11–0.15, and 0.1 s of delay jitter already costs 0.14. The timing it needs is sub-second. The data-grounded reading is band FC (`fit/band_fc.py`, October): the data's FC is the same in every band below 0.2 Hz, and the 3x solve carried its FC in its fast bands, so A1 on a band-matched solve (`--fit-bands`) is the check that needs no haemodynamic model. 1x **done**: controls 0.07–0.11 against 0.723; tolerates more jitter (0.1 s costs 0.024, floor at ~2 s). 5x **done**: controls 0.18–0.21 against 0.739; 0.1 s costs 0.15. Map-weighted **done**: controls 0.08–0.13 against 0.739; 0.1 s costs 0.09 |
| A2 | medium-only phase spread over the reach against cycles per decay | whether fτ = ½ (phase π across one attenuation length) explains the switch | **done**: no. The in-phase crossing moves with reach relative to the sheet (0.19 on a plane, 0.36 at 74 mm, 0.51 at ~220 mm, 1.27 at 664 mm), and the fitted switches sit within 0.03 of their media's crossing. See [phaseA_findings.md](phaseA_findings.md) |
| A3 | two media at the same decay but different speed, so reach differs from hemisphere size | medium alone vs medium with sheet size | **done** ([findings](phaseA_findings.md#a3-on-fitted-drives)). The in-phase crossing is an upper limit: the fitted switch follows it at 664 mm (1.29 against 1.27), but at 74 mm the fit cancels at every frequency. 664 mm fit unconverged at 150 evaluations; a 600-evaluation check is queued |
| A4 | refit 3x uniform through the HRF | whether the fitted timing survives a haemodynamic observation stage | **dropped** (see Decisions). The two fits that were run are kept as a record ([findings](phaseA_findings.md#a4-the-3x-fit-through-the-hrf-open-needs-a-decision-on-the-observation-model)). Through the HRF alone the fit becomes 97% one global signal (score 0.669), which double-centred Spearman cannot see. Needs the data's detrending and global-signal removal in the observable, and a global-share check beside every score, before A4 can be read |
| A5 | categories on two or three equally good solutions of one medium | switch belongs to the FC or to the solver | |
| A6 | designed gradings at fixed strengths (one map per role, e.g. +-0.15 and +-0.3 per sd); optional held-out check of one or two weights | what heterogeneity does; whether any grading is justified | |

## Phase B: fix the configuration and triage

Configuration decisions, to settle with colleagues:

| decision | candidates |
|---|---|
| reference medium | uniform, with speed, decay and heterogeneity as designed variations (decided) |
| solver and stopping | factored rank 20 at 150 evaluations, or projected gradient; stopping rule validated on held-out subjects |
| scoring | realisation length, draws, pooled vs per-draw, metrics (Spearman plus one that sees magnitude), 2,000 solve vertices |
| input basis | 100 ascending first-order regions |

A result is kept if (1) it carries a step of the argument, (2) it survives the current code
and the fixed configuration, (3) it is robust to the choices FC cannot decide, or its
dependence on them is the point, and (4) a control says what produces it.

First-pass triage:

- **Core:** fit against the ceiling, with undriven cortex; the input ladder (white everywhere,
  white in territory, independent, random coherent, timing only, fitted) across media; FC
  constrains without identifying; how input drives waves (interference, steering, the switch,
  controls); what BOLD sees depends on decay and observation model.
- **Supporting:** checkerboard (rerun; it used the old 47-piece medium); lead-lag invisible to
  zero-lag FC; FC within frequency bands; the linear/nonlinear tie; the envelope observable.
- **Methods:** the convex formulation and its exactness checks; the clock; matching the
  measurement; stopping as regularisation; Spearman's blindness to magnitude.
- **Limitations:** over-smooth model FC, the 10-30 mm dip, dynamic range, checkerboard
  amplitude, geometry not yet varied.
- **Dropped (history only):** genetic algorithm; old-clock and nilearn-target results;
  sensory, DMN and spread bases (possibly one territory comparison kept as a rung of the
  ladder); tractography basis; whitening; affinity scoring; map search; switching media;
  structural coupling; MSC; denoising.

## Phase C: the rerun

1. Merge `claude/sweet-ramanujan-xrccxd` into `main`, with the unfiltered target replacing
   `main`'s bandpassed one; run the tests; tag the commit the reruns use.
2. One configuration preset that every script reads.
3. Results written to durable storage (the shared Dropbox folder) as each stage finishes.
4. Order: target and ceiling; impulse responses per medium; solves per medium; scoring;
   input ladder; wave analyses; supporting analyses; figures.
5. Each stage reproduces a known anchor number before its new numbers are trusted.
6. One numbers ledger, generated from the files.

## Phase D: presentation

Toy examples first (see Decisions). Then one figure per step of the argument, provisionally:
the idea and the ascending territory; the fit, with undriven cortex; the input ladder; equally
good mechanisms; how input drives waves; the switch across media; what BOLD sees; the
checkerboard. Colleagues see the open configuration decisions and the Phase A results; the
paper waits for Phase C.
