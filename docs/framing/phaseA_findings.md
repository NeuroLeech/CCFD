# Phase A findings

Results of the decision experiments in [plan.md](plan.md), recorded as they come in.
Every number here was computed in this rerun, on the code at the commit that adds it.

---

## A2 and the medium-only part of A3: where in-phase inputs stop adding

**Question.** The fitted drives switch from reinforcing to cancelling at about half a
cycle per decay time in all four wave-analysis media (docs/waves-findings.md: 0.54, 0.53,
0.51, 0.42). Is that switch a property of the medium alone, as the "phase advances by π
over one attenuation length at fτ = ½" argument says? Or does it depend on how the medium's
reach compares with the size of the sheet?

**Test.** No fitted drive is used. Every one of the 100 input pieces is driven at unit
amplitude and in phase. At each of 150 whole-cortex patches,
R = |Σ_k h_k| / Σ_k |h_k|. R is 1 when all contributions arrive in phase, and falls as
their propagation phases spread. The media are:

- the four recorded media;
- three flat stand-ins with the decay held at 8.33 s and only the speed changed, so that
  reach changes and fτ does not;
- an infinite flat plane, computed analytically (`phase_reach.plane_R`). It has no scale
  except the decay, so its curve depends on fτ alone.

`analysis/phase_reach.py`, about 2 hours single-threaded.

![A2](figures/phaseA2_reach.png)

### Results

**1. The stand-ins are a valid tool.** The stand-in built at 8.33 s and a spread label
of 4.5 reproduces the recorded 3x medium to every printed digit.

**2. The spread label is not the wave speed.** On the sheet the model runs on, waves travel
at 5.9× the spread label. The label is an older time-to-peak spread measured on the white
surface (`core/timescale.py`, `core/units.py`). The true speed is read off the
high-frequency phase speed:

| medium | label | wave speed | reach (speed × decay) |
|---|---|---|---|
| 1x | 1.5 mm/s | 8.8 mm/s | 220 mm |
| 3x | 4.5 mm/s | 26.5 mm/s | 221 mm |
| 5x | 7.5 mm/s | 44 mm/s | 222 mm |

For scale: the geodesic distances between the 150 patch centres have a median of 119 mm and
a maximum of 257 mm. So all four recorded media carry a response about as far as the
sheet is wide. Any "reach" quoted from the labels (e.g. 4.5 × 8.33 = 37.5 mm) understates
it about sixfold.

**3. The crossing moves with reach.** The fτ at which median R falls through 0.5,
interpolated:

| medium | reach | crossing (cycles per decay) |
|---|---|---|
| infinite flat plane | none (no sheet) | 0.19 |
| stand-in, 8.33 s | 74 mm | 0.36 (R hovers at 0.50–0.52 from 0.29 to 0.46) |
| 1x, 3x, 5x | ~220 mm | 0.51, 0.51, 0.51 |
| 3x map-weighted | ~220 mm (median) | 0.45 |
| stand-in, 8.33 s | 664 mm | 1.27 |

With the decay fixed, tripling the reach moves the crossing from 0.51 to 1.27 cycles per
decay. A third of the reach moves it down to 0.36. The analytic plane, the limit of a
sheet much larger than the reach, sits at 0.19.

So the frequency at which in-phase inputs stop adding is **not set by the medium alone**.
It depends on reach relative to the sheet:

- When the reach is short, the decay limits it, and it tends towards the plane's 0.19.
- When the reach is long, the sheet limits it: the spread of travel distances among the
  inputs is capped by the sheet, so it takes a higher frequency to spread their phases.

The recorded media sit where these two limits meet. That is why they line up at half a
cycle per decay.

**4. The fitted switch follows the medium's crossing.** Per medium:

| medium | in-phase crossing (medium only) | fitted switch (waves-findings) |
|---|---|---|
| 1x | 0.51 | 0.54 |
| 3x | 0.51 | 0.53 |
| 5x | 0.51 | 0.51 |
| 3x map-weighted | 0.45 | 0.42 |

The fitted switch comes out within 0.03 of the in-phase crossing, and the map-weighted medium is lowest
on both. That 0.03 is below the resolution of the fitted switch, which is read from bins
a factor of about 1.45 apart (see Rebuilds below). Only two distinct values of the medium-only crossing are in play, so this is
agreement, not a correlation. The fitted values are from the impulse-fixed solves; they will
be replaced by the rebuilt solves' values when the queue reaches them.

**5. The derivation in waves-findings is withdrawn as the explanation.** "Phase π across
one attenuation length at fτ = ½" gives the right number only because reach ≈ sheet in
those media. On the plane the same in-phase sum stops adding at 0.19, near
1/(2π) ≈ 0.16, where the uniform response |1/(1/τ − iω)|² rolls off. The medium law itself
holds where it can be tested. Write k for the complex wavenumber, ω = 2πf for the angular
frequency and c for the wave speed:

- **The law, k = (ω + i/τ)/c.** For the 74 mm stand-in at high fτ, the mesh matches the
  flat-plane Green's function. At fτ = 5 the phase/amplitude slope ratio is 31.7 against
  2πfτ = 31.4, and the attenuation length is 74 mm against cτ = 73 mm.
- **Where the check cannot be made.** The dispersion check is uninformative for the
  ~220 mm-reach media. Its 15–60 mm annuli lie in the near field, where even the
  flat-plane Green's function does not give the far-field ratio.

**6. Two smaller observations.**

- **Timing patterns.** Count the distinct timing patterns the patches ask for, relative to
  the count without the phases. At half a cycle per decay it is about 2.7× in the recorded
  media and about 3× at a 74 mm reach. At a 664 mm reach it stays near 1 up to about 0.7
  cycles per decay.
- **Resonance.** Around fτ ≈ 0.4–0.45 the ~220 mm media's amplitude barely falls with
  distance (the L column in the log). This is where the sheet's lowest modes sit: the
  wavelength there is about 480 mm, roughly twice the largest patch distance. In these media, that resonance and the crossing
  land at about the same frequency.

### What this changes

- **Framing point 3**, that medium, geometry and input cannot be separated from FC alone,
  now has a fit-free, quantitative instance. The frequency below which inputs must be
  timed to reinforce, and above which the waves are what is left after cancellation, is
  set by the medium and the sheet together. The input timing then supplies the contrast
  either side of it (the controls in waves-findings).
- **The BOLD-band conclusions.** In waves-findings, BOLD sees mostly reinforcing waves in
  3x and 5x and mostly cancelling ones in 1x. Whether that is so depends on where the
  crossing falls relative to the BOLD band, so it depends on reach relative to sheet, not
  on the decay alone. A 664 mm reach at 8.33 s puts the in-phase crossing at 0.15 Hz,
  above the whole 0.01–0.08 Hz band.
- **Toy examples.** "Phased array on a finite sheet" is the right toy: the same timing
  rule reinforces or cancels depending on f, on the decay and on the travel time across
  the sheet.

### Test queued after A2 (results under "A3 on fitted drives" below)

Two solves at decay 8.33 s with reach 74 mm and 664 mm (`nobp_sm6_reach033_box`,
`nobp_sm6_reach3_box`), with `interplay.py bold` and its controls on each, and
`interplay.py media` across them and the rebuilt 3x. **Prediction:** the fitted switch
moves with the in-phase crossing, to about 0.36 cycles per decay (0.043 Hz) and about 1.27
(0.15 Hz). If instead it stays near 0.5, the switch belongs to something in the fit rather
than to the medium on this sheet.

### A3 on fitted drives

Three solves at the same decay (8.33 s), differing only in reach: `nobp_sm6_reach033_box`
(spread label 1.5), the rebuilt 3x `nobp_sm6_s3_box` (4.5) and `nobp_sm6_reach3_box` (13.5).
Each has `interplay.py bold` under the fitted drive and its six controls.

![A3](figures/phaseA3_fit_reach.png)

The curve in each panel is the share of the 150 patches reinforcing minus the share
cancelling, under the fitted drive. Above zero, more patches build their waves by
reinforcing; below, by cancelling. Grey is the range over the six controls (random coherent
drive and timing-only control, three seeds each). The dotted line is the medium-only
in-phase crossing from A2. The beige band is 0.01–0.08 Hz.

| | 74 mm | 221 mm (3x, rebuilt) | 664 mm |
|---|---|---|---|
| in-phase crossing (A2, medium only) | 0.36 | 0.51 | 1.27 |
| fitted switch | **none** | 0.53 | **1.29** (0.155 Hz) |
| fit (2,000 solve vertices) | 0.7777 | 0.8087 | 0.6360 |
| realised, 4 draws | +0.7080 ± 0.0064 | +0.7568 ± 0.0014 | +0.6027 ± 0.0026 |
| gap, fit to realised | 0.057 | 0.070 | 0.103 |
| rank | 28.7 | 18.6 | 2.2 |
| still gaining at 150 evaluations (last 15) | 0.0006 | 0.0011 | 0.0070 |
| reinforcing / cancelling, HRF + 0.01–0.08 Hz | 5% / 58% | 53% / 29% | 66% / 15% |

**Long reach: the prediction held.** The fitted drive reinforces at 46–91% of patches from
0.13 to 1.02 cycles per decay and cancels at 59–99% from 1.46 up. The switch between them,
interpolated, is at 1.29 cycles per decay against a predicted 1.27. The bins there are a
factor of 1.43 apart, so this says the switch lies between 1.02 and 1.46, which contains
the prediction, and is far from the 0.5 of the recorded media.

One qualification about how the switch is read. `interplay.py media` reports the *first*
bin where cancelling overtakes reinforcing. In the 664 mm solve that is a flicker at 0.08
cycles per decay (27% reinforcing against 35% cancelling, with 27% single source), after
which reinforcing leads again until 1.29. I added a second readout, the crossing above
which cancelling leads in every bin, after seeing this solve. For the 3x solve the two
readouts agree (0.53). The figure uses the second.

**Short reach: the prediction failed** (left panel). There is
no reinforcing regime: cancelling leads in every bin, at 57–87% of patches. In-phase
inputs could still add below 0.36 cycles per decay (in-phase R 0.99 at 0.04), but the
fitted drive does not use that.

**The controls.** Comparing the fitted curve with the range of the six controls, bin by bin:

- **Above its switch**, the fit cancels more than any control in every bin, in both the
  221 mm (6 of 6 bins) and 664 mm (4 of 4) media.
- **Below its switch**, the fit reinforces more than any control in 5 of 8 bins (221 mm)
  and 4 of 10 (664 mm). In the rest, which include the lowest bins in both media, it lies
  inside the range.
- **In the 74 mm medium** it cancels more than any control in 13 of the 14 bins.

The controls carry a lean of their own, set by the medium. In the 664 mm medium their
mean is +0.27 to +0.56 (towards reinforcing) from 0.13 to 1.02 cycles per decay, and two
of the three timing-only seeds never cross over. In the 74 mm medium their mean is −0.14
to −0.33 at every frequency. In the 221 mm medium it stays between −0.21 and +0.10. So the
medium sets a lean, and the fit works with it or against it:

- At 74 mm the fit goes further in the medium's direction (cancelling) at every frequency.
- At 664 mm it goes further in the medium's direction (reinforcing) below the switch, and
  reverses it above.
- At 221 mm the controls barely lean, so both regimes come from the fit.

**What A3 says.** Taking both reaches together:

- The in-phase crossing behaves as an **upper limit** on where reinforcing can lead. In
  none of the three media does reinforcing lead above it.
- Whether reinforcing leads **below** it is not fixed by the crossing. In two of the three
  media the fit uses reinforcing all the way up to the crossing. In the 74 mm medium it
  never does.
- So the switch the waves analysis found at half a cycle per decay is the medium-and-sheet
  limit, reached by a fit that chooses to reinforce. Both parts are needed: the limit from
  the medium and sheet, the choice from what the FC asks of that medium.

**The 664 mm fit is not converged.** At 150 evaluations it was still gaining about 0.007
per 15 evaluations, six times the 3x rate. Its effective rank of 2.2 and its realised score
are therefore from a fit stopped early, and the fit scores in the table are not a
like-for-like comparison of the three media. The 150-evaluation budget is the project
convention and stands in for a realisability penalty (see `--prank-mu` in
`fit/best_fit.py`), but that brake clearly binds harder in this medium. A 600-evaluation solve
(`nobp_sm6_reach3_f600_box`) is queued after the rebuilds to check whether the switch
stays at about 1.3 as the fit converges.

Logs: [build_reach033](logs/build_reach033.txt), [bold_reach033](logs/bold_reach033_box_fitted.txt),
[build_reach3](logs/build_reach3.txt), [bold_reach3](logs/bold_reach3_box_fitted.txt),
[media_reach](logs/media_reach.txt).

### Reproduce

```bash
python analysis/phase_reach.py --tags nobp_sm6_s1,nobp_sm6_s3,nobp_sm6_s5,nobp_sm6_mapsmm_s3 \
  --standin 8.3333,1.5 --standin 8.3333,4.5 --standin 8.3333,13.5
python analysis/phase_reach.py --figure results/phase_reach_<...>.npz    # the A2 figure
python analysis/phase_reach.py --fit-figure results/phase_reach_<...>.npz  # the A3 figure (needs the bold caches)
```

---

## Rebuilds of the wave-analysis solves

Each rebuild is held to its recorded values within draw scatter. Recorded values are from
docs/waves-findings.md, at 150 evaluations.

| solve | fit (2,000 solve vertices) | realised, 4 draws of 577 s | rank | recorded | verdict |
|---|---|---|---|---|---|
| `nobp_sm6_s3_box` (3x flat) | 0.8087 | +0.7568 ± 0.0014 | 18.6 | fit 0.809, realised 0.756 ± 0.002, rank 18.6 | reproduced |

Log: [logs/build_s3.txt](logs/build_s3.txt). The 3x build took 28 minutes in this container.

**Categories and switch, 3x rebuilt** (`interplay.py bold` and `media`). Shares are
single source / reinforcing / cancelling / other, %, over all patches:

| reading | rebuilt | recorded |
|---|---|---|
| raw field | 5 / 4 / 77 / 14 | 5 / 5 / 76 / 15 |
| this model's observable | 5 / 11 / 74 / 10 | 5 / 11 / 73 / 10 |
| HRF | 8 / 43 / 37 / 12 | 8 / 43 / 37 / 12 |
| HRF + 0.01–0.08 Hz | 7 / 53 / 29 / 11 | 7 / 53 / 29 / 11 |
| switch, reinforcing to cancelling | 0.063 Hz = 0.53 cycles per decay | 0.063 Hz = 0.53 |

The timing-only control's first crossovers come out at 0.13, 0.11 and 0.04 cycles per
decay for the three seeds, inside the recorded 0.03–0.85 range.

**Resolution of the switch.** The switch is read from 14 log-spaced bins. Near it they
sit at 0.34, 0.49 and 0.71 cycles per decay, a factor of about 1.45 apart. So differences
of a few hundredths between a fitted switch and an in-phase crossing (A2, finding 4) are
below what the bins resolve. The A3 prediction is a shift of about a factor of 0.7 or
2.5, which they resolve easily.
