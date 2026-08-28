"""Score a model run against the MSC vertexwise FC, cheaply enough to sit in a search loop.

Everything that does not depend on the candidate is done once when the target is built:
the vertex intersection between the FC matrix and the Cortex submesh, the edge sample,
the target FC values on those edges, and their ranks. A candidate then costs one rank
transform of (V x T) and one pass over the sampled edges - no V x V matrix is ever formed.

    from mesh_cache import load_cortex
    from fc_score import FCTarget
    cortex = load_cortex("fsaverage5")
    target = FCTarget(cortex)                    # ~2 s, then reuse for every candidate
    r = target.score(frames)                     # ~0.2 s per run

Edges are sampled once with a fixed seed, so every candidate is scored on the same edge
set and scores are comparable across a search. 2e6 of the ~44e6 edges puts the sampling
error on r near 7e-4; pass n_edges=None to use all of them (Pearson only).

Vertices that never move in a given run are given zero coupling with everything rather
than being dropped, so the edge set does not change from candidate to candidate. How many
there were is reported by score(..., report=True).
"""
import os, re, glob
import numpy as np
from scipy.stats import rankdata

from fc_vertexwise import align_to_cortex, FCDIR


def default_fc(space, mask="glasser", subject="*", outdir=FCDIR):
    """Most recently written FC matrix for this space/mask, whichever dataset it came
    from - pass subject='sub-MSC01' or a full fc_path to pin one."""
    pat = os.path.join(outdir, f"{subject}_hemi-L_space-{space}_mask-{mask}*fc.npy")
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(f"no FC matrix matching {pat} - run fc_vertexwise.py first")
    return hits[-1]


def raw_fc(space, mask="glasser", subject="*", outdir=FCDIR):
    """Newest UN-centred FC matrix for this space/mask.

    default_fc matches both `...spearmanfc.npy` and `...spearmandcfc.npy` and takes
    whichever was written last, which is how the pre-centred file came to be the default
    target. That matters because a pre-centred target is scored against an un-centred
    model: the centring then applies to one side only."""
    pat = os.path.join(outdir, f"{subject}_hemi-L_space-{space}_mask-{mask}*fc.npy")
    hits = [h for h in sorted(glob.glob(pat), key=os.path.getmtime)
            if not os.path.basename(h).endswith("dcfc.npy")]
    if not hits:
        raise FileNotFoundError(f"no un-centred FC matrix matching {pat}")
    return hits[-1]


def default_target(cortex, centre="double", **kw):
    """FCTarget that centres the TARGET and the MODEL the same way.

    With centre='double' the raw matrix is loaded and double-centred here, and
    FCTarget.model_edges applies the same operation to the model - so both sides have
    had their global component removed. Loading the pre-centred file with centre='none'
    instead leaves the model un-centred, which is worth about 0.02 of score and, more to
    the point, distorts diagnostics: it makes the model look as though it cannot produce
    anticorrelation at all, when symmetrically centred it is 56% negative against the
    target's 61%."""
    path = raw_fc(cortex.mesh) if centre == "double" else default_fc(cortex.mesh)
    return FCTarget(cortex, fc_path=path, centre=centre, **kw)


def double_centre(FC, inplace=False):
    """Subtract row and column means, add the grand mean, all excluding the diagonal.

    An FC matrix built from data without global signal regression is dominated by a
    global component: every vertex correlates positively with every other, so an edgewise
    score mostly measures that offset. Double-centring removes the additive part of it -
    the linear analogue of GSR, which removes the same component multiplicatively (GSR
    yields partial correlations, dividing by sqrt(1 - r_ig^2) as well as subtracting).

    The diagonal is excluded from the means and left at 1, so a vertex's own value cannot
    inflate its row mean."""
    n = FC.shape[0]
    A = FC if inplace else FC.astype(np.float32, copy=True)
    d = np.diag(A).copy()
    rows = (A.sum(1) - d) / (n - 1)
    grand = float((rows.sum() * (n - 1)) / (n * (n - 1)))
    A -= rows[:, None]
    A -= rows[None, :]
    A += grand
    np.fill_diagonal(A, 1.0)
    return A


def vertices_path(fc_path):
    """Companion vertex-id file for any of the fc naming variants."""
    return re.sub(r"_(spearman|pearson)(dc)?fc\.npy$", "_vertices.npy", fc_path)


def _rank_z(X):
    """Rows -> rank transformed, zero mean, unit variance. Flat rows come back all zero,
    which makes their correlation with everything exactly 0."""
    R = rankdata(X, axis=1).astype(np.float32)
    R -= R.mean(1, keepdims=True)
    sd = R.std(1, keepdims=True)
    flat = (sd == 0).ravel()
    sd[flat] = 1.0
    R /= sd
    R[flat] = 0.0
    return R, flat


class FCTarget:
    """Empirical FC on the vertices a model run actually produces, plus a scoring function.

    cortex  : Cortex from mesh_cache.load_cortex, the mesh the run was produced on
    fc_path : FC .npy; defaults to the newest glasser-mask matrix for cortex.mesh
    burn    : model frames to discard before correlating
    n_edges : size of the fixed edge sample (None = every edge, Pearson only)
    metric  : 'spearman' or 'pearson' comparison between the two edge vectors
    """

    def __init__(self, cortex, fc_path=None, burn=50, n_edges=2_000_000, seed=0,
                 metric="spearman", centre="none", verbose=True):
        if metric not in ("spearman", "pearson"):
            raise ValueError(metric)
        if centre not in ("none", "double"):
            raise ValueError(centre)
        self.centre = centre
        self.moran, self.moran_lambda, self.moran_tol = None, 0.0, 0.0
        self.rank_min, self.rank_lambda = 10.0, 0.0
        if n_edges is None and metric == "spearman":
            raise ValueError("metric='spearman' needs an edge sample; pass n_edges")
        self.fc_path = fc_path or default_fc(cortex.mesh)
        self.burn, self.metric, self.n_edges, self.seed = burn, metric, n_edges, seed

        fc = np.load(self.fc_path, mmap_mode="r")
        vertices = np.load(vertices_path(self.fc_path))
        fc, cols, ids = align_to_cortex(np.asarray(fc), vertices, cortex)
        self.cols, self.vertices, self.nV = cols, ids, fc.shape[0]
        dropped = cortex.nV - self.nV
        if centre == "double":
            fc = double_centre(fc, inplace=True)

        if n_edges is None:
            self.i = self.j = None
            iu = np.triu_indices(self.nV, 1)
            y = fc[iu].astype(np.float32)
            self.fc = fc                                  # kept for the full-matrix path
        else:
            rng = np.random.default_rng(seed)
            i = rng.integers(0, self.nV, n_edges, dtype=np.int64)
            j = rng.integers(0, self.nV, n_edges, dtype=np.int64)
            keep = i != j
            i, j = i[keep], j[keep]
            lo, hi = np.minimum(i, j), np.maximum(i, j)   # one orientation per edge
            self.i, self.j = lo.astype(np.int32), hi.astype(np.int32)
            y = np.asarray(fc[self.i, self.j], np.float32)
            self.fc = None
        del fc

        self.y = self._prep(y)
        if verbose:
            print(f"[FCTarget] {os.path.basename(self.fc_path)}\n"
                  f"           {self.nV} vertices ({dropped} of cortex.nV unmatched), "
                  f"{len(self.y) if n_edges else self.nV*(self.nV-1)//2} edges, "
                  f"metric={metric}, centre={centre}, burn={burn}")

    # -------------------------------------------------------------- internals
    def _prep(self, v):
        """Rank (if Spearman) and z-score an edge vector, so scoring is one dot product."""
        v = rankdata(v).astype(np.float32) if self.metric == "spearman" else np.asarray(v, np.float32)
        v = v - v.mean()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def model_z(self, frames):
        """Rank z-scored model rows (V, T) - the one thing every model quantity needs."""
        return _rank_z(np.ascontiguousarray(frames[self.burn:, self.cols].T))

    def model_edges(self, frames=None, chunk=250_000, Z=None, flat=None):
        """Model Spearman FC evaluated only on the sampled edges -> (n_edges,) float32."""
        if Z is None:
            Z, flat = self.model_z(frames)
        T, V = Z.shape[1], Z.shape[0]
        if self.i is None:
            S = (Z @ Z.T) / T
            if self.centre == "double":
                S = double_centre(S, inplace=True)
            iu = np.triu_indices(self.nV, 1)
            return S[iu], flat
        out = np.empty(len(self.i), np.float32)
        for a in range(0, len(self.i), chunk):
            b = slice(a, a + chunk)
            out[b] = np.einsum("ij,ij->i", Z[self.i[b]], Z[self.j[b]]) / T
        if self.centre == "double":
            # row mean of a correlation matrix is a dot product with the summed rows,
            # so centring costs O(V*T) rather than the V x V matrix it describes
            ssum = Z.sum(0)
            diag = (Z * Z).sum(1) / T                      # 1, or 0 for a flat vertex
            m = ((Z @ ssum) / T - diag) / (V - 1)
            grand = (float(ssum @ ssum) / T - float(diag.sum())) / (V * (V - 1))
            out -= m[self.i]; out -= m[self.j]; out += grand
        return out, flat

    # -------------------------------------------------------------- scoring
    def attach_moran(self, matcher, lam=1.0, tol=0.0):
        """Add a spatial-scale term to the fitness. `matcher` needs .gap(Z) -> float,
        the mean |Moran's I difference| between this run and the empirical FC across
        rings; see fc_moran.MoranMatch.

        Fitness is similarity - lam * max(0, gap - tol). Inside `tol` the run is not
        penalised at all: the empirical correlogram is not a precise number, and its own
        wobble sets the scale - 0.0025 between two 50-subject halves of NKI, 0.018 for
        the same subjects smoothed 4 mm further, 0.048 at 8 mm. A tolerance near 0.02
        therefore means 'as close as two defensible pipelines are to each other', and
        only excursions beyond that cost anything."""
        self.moran, self.moran_lambda, self.moran_tol = matcher, float(lam), float(tol)
        return self

    def attach_rank(self, min_rank=10.0, lam=1.0):
        """Require the field to keep at least `min_rank` effective dimensions.

        Both high-scoring runs so far reached their score by collapsing the field: the
        slow-drive winner to effective rank 4.3, the damped-medium winner to 1.7, against
        95 for the empirical FC. Moran's I does not see this - a rank-2 field can have a
        perfectly ordinary correlogram - so rank needs its own term. Effective rank is the
        participation ratio of the field's singular values, and the penalty is
        lam * max(0, (min_rank - rank) / min_rank), zero once the floor is cleared."""
        self.rank_min, self.rank_lambda = float(min_rank), float(lam)
        return self

    @staticmethod
    def effective_rank(frames):
        """Participation ratio of the temporal covariance - cheap and identical to the
        spatial one, since both share the same nonzero spectrum."""
        X = np.asarray(frames, np.float64)
        X = X - X.mean(0, keepdims=True)
        ev = np.linalg.eigvalsh(X @ X.T)               # (T, T), T << nV
        ev = np.clip(ev, 0.0, None)
        return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))

    def score(self, frames, report=False):
        """-> similarity between the run's FC and the empirical FC over the fixed edge
        set, minus the Moran penalty if one is attached."""
        Z, flat = self.model_z(frames)
        x, _ = self.model_edges(Z=Z, flat=flat)
        sim = float(self._prep(x) @ self.y)
        gap = self.moran.gap(Z) if getattr(self, "moran", None) is not None else 0.0
        excess = max(0.0, gap - getattr(self, "moran_tol", 0.0))
        total = sim - getattr(self, "moran_lambda", 0.0) * excess
        rank = np.nan
        if getattr(self, "rank_lambda", 0.0):
            rank = self.effective_rank(frames[self.burn:])
            total -= self.rank_lambda * max(0.0, (self.rank_min - rank) / self.rank_min)
        if report:
            return total, dict(flat_vertices=int(flat.sum()), n_edges=len(x),
                               frames_used=frames.shape[0] - self.burn,
                               similarity=sim, moran_gap=float(gap),
                               moran_excess=float(excess), rank=float(rank))
        return total

    def model_fc(self, frames):
        """Full (nV, nV) model Spearman FC on the aligned vertices, for inspection."""
        M = np.ascontiguousarray(frames[self.burn:, self.cols].T)
        Z, _ = _rank_z(M)
        S = (Z @ Z.T) / Z.shape[1]
        np.clip(S, -1.0, 1.0, out=S)
        np.fill_diagonal(S, 1.0)
        return double_centre(S, inplace=True) if self.centre == "double" else S

    def target_fc(self):
        """Full (nV, nV) empirical FC on the aligned vertices."""
        if self.fc is not None:
            return self.fc
        fc = np.load(self.fc_path, mmap_mode="r")
        vertices = np.load(vertices_path(self.fc_path))
        idx = np.searchsorted(vertices, self.vertices)
        fc = np.asarray(fc)[np.ix_(idx, idx)]
        return double_centre(fc, inplace=True) if self.centre == "double" else fc


if __name__ == "__main__":
    import time, sys
    from mesh_cache import load_cortex

    frames_path = sys.argv[1] if len(sys.argv) > 1 else "results/frames.npy"
    frames = np.load(frames_path)
    cortex = load_cortex("fsaverage5", verbose=False)

    t0 = time.time(); target = FCTarget(cortex); build = time.time() - t0
    t0 = time.time(); r, info = target.score(frames, report=True); s = time.time() - t0
    print(f"  {os.path.basename(frames_path)}: {frames.shape[0]} frames, "
          f"{info['frames_used']} used, {info['flat_vertices']} flat vertices")
    print(f"  score {r:+.4f}   (target built in {build:.1f}s, scored in {s:.2f}s)")
