"""How much of the target FC is real? Split-half reliability, in the units of the score.

The model is scored as Spearman between its edge values and the group target's, over a
fixed 2e6 edge sample. That number is bounded above by the target's own reliability: even
a perfect model can only reach corr(true FC, our 99-subject estimate).

So: split the subjects in half, build a group FC from each half exactly as fc_group_nki
does (Spearman per subject, Fisher z, tanh of the mean), and score one half against the
other on the target's own edge sample. Spearman-Brown converts the reliability of a half
into the reliability of the whole, and its square root is the ceiling on the score.

The halves are built the way the target was - including the double centring, which
removes the global component and so removes the most reliable part of the matrix. A
reliability measured before that step answers a question about a matrix nobody fits.

  python reliability.py --splits 8
"""
import argparse, time
import numpy as np

from mesh_cache import load_cortex
import fc_score
import fc_group_nki as nki
import holdout


def spearman_brown(r, n_from, n_to):
    """Reliability of an n_to-subject average, given r for an n_from-subject one."""
    m = n_to / n_from
    return m * r / (1.0 + (m - 1.0) * r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", type=int, default=3,
                    help="random half-splits (each is a full pass over the subjects)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    c = load_cortex("fsaverage5", verbose=False)
    t = fc_score.default_target(c, verbose=True)
    n = len(nki.subject_files("left"))

    rhos = []
    for k in range(a.splits):
        e, _, _ = holdout.half_targets(t, seed=a.seed + k, verbose=(k == 0))
        ya, yb = t._prep(e[0]), t._prep(e[1])
        rhos.append(float(ya @ yb))
        print(f"  split {k+1}: spearman between halves {rhos[-1]:+.4f}  "
              f"(each half vs the 99-subject target: {float(ya @ t.y):+.4f} / "
              f"{float(yb @ t.y):+.4f})", flush=True)

    r_half = float(np.mean(rhos))
    rel = spearman_brown(r_half, n // 2, n)
    print(f"\n  half ({n//2} subjects) reliability {r_half:+.4f} +- {np.std(rhos):.4f}")
    print(f"  whole ({n} subjects) reliability {rel:+.4f} (Spearman-Brown)")
    print(f"  ceiling on the score, corr(true FC, our target) = sqrt(rel) = "
          f"{np.sqrt(rel):+.4f}")
    print(f"\n  measured on the double-centred matrix the model is actually scored "
          f"against, over the target's own {len(t.i)} sampled edges")


if __name__ == "__main__":
    main()
