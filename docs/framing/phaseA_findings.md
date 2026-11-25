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
on both. Only two distinct values of the medium-only crossing are in play, so this is
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

### Test now queued

Two solves at decay 8.33 s with reach 74 mm and 664 mm (`nobp_sm6_reach033_box`,
`nobp_sm6_reach3_box`), with `interplay.py bold` and its controls on each, and
`interplay.py media` across them and the rebuilt 3x. **Prediction:** the fitted switch
moves with the in-phase crossing, to about 0.36 cycles per decay (0.043 Hz) and about 1.27
(0.15 Hz). If instead it stays near 0.5, the switch belongs to something in the fit rather
than to the medium on this sheet.

### Reproduce

```bash
python analysis/phase_reach.py --tags nobp_sm6_s1,nobp_sm6_s3,nobp_sm6_s5,nobp_sm6_mapsmm_s3 \
  --standin 8.3333,1.5 --standin 8.3333,4.5 --standin 8.3333,13.5
python analysis/phase_reach.py --figure results/phase_reach_<...>.npz    # the figure above
```
