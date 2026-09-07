"""Intrinsic Delaunay flipping with incremental connectivity updates (O(1) per flip)."""
import numpy as np
from collections import defaultdict

def _ang(a, b, c):
    return np.arccos(np.clip((b*b + c*c - a*a) / (2*b*c), -1.0, 1.0))

def key(i, j):
    return (i, j) if i < j else (j, i)

def intrinsic_delaunay(F, L0, max_flips=2_000_000):
    F = [list(map(int, t)) for t in F]
    Lg = dict(L0)
    e2t = defaultdict(list)
    for t, (a, b, c) in enumerate(F):
        e2t[key(b, c)].append((t, 0)); e2t[key(c, a)].append((t, 1)); e2t[key(a, b)].append((t, 2))

    def opp_angle(t, corner):
        v = F[t]
        a = Lg[key(v[(corner+1) % 3], v[(corner+2) % 3])]
        b = Lg[key(v[corner], v[(corner+2) % 3])]
        c = Lg[key(v[corner], v[(corner+1) % 3])]
        return _ang(a, b, c)

    def is_del(e):
        ts = e2t.get(e, [])
        if len(ts) != 2: return True
        return opp_angle(*ts[0]) + opp_angle(*ts[1]) <= np.pi + 1e-12

    stack = [e for e in list(e2t) if not is_del(e)]
    instack = set(stack)
    nflip = 0
    while stack and nflip < max_flips:
        e = stack.pop(); instack.discard(e)
        ts = e2t.get(e, [])
        if len(ts) != 2 or is_del(e): continue
        (t1, c1), (t2, c2) = ts
        i, j = e
        k = F[t1][c1]; l = F[t2][c2]
        if k == l: continue
        kl = key(k, l)
        if kl in Lg: continue                       # would duplicate an edge

        def ang_at(vert, o1, o2):
            return _ang(Lg[key(o1, o2)], Lg[key(vert, o2)], Lg[key(vert, o1)])
        ang_i = ang_at(i, j, k) + ang_at(i, j, l)
        if ang_i >= np.pi - 1e-12: continue
        ik, il = Lg[key(i, k)], Lg[key(i, l)]
        newlen = np.sqrt(max(ik*ik + il*il - 2*ik*il*np.cos(ang_i), 1e-18))

        around = [key(i, k), key(k, j), key(j, l), key(l, i)]
        for ee in around + [e]:
            if ee in e2t:
                e2t[ee] = [(t, c) for (t, c) in e2t[ee] if t not in (t1, t2)]
        F[t1] = [k, l, j]; F[t2] = [l, k, i]
        del Lg[e]; Lg[kl] = newlen
        for t in (t1, t2):
            a, b, c = F[t]
            e2t[key(b, c)].append((t, 0)); e2t[key(c, a)].append((t, 1)); e2t[key(a, b)].append((t, 2))
        if not e2t[e]: del e2t[e]
        nflip += 1
        for ee in around + [kl]:
            if ee in e2t and not is_del(ee) and ee not in instack:
                stack.append(ee); instack.add(ee)
    return np.array(F), Lg, nflip


def cotan_from_lengths(F, Lg):
    edges = sorted(Lg.keys())
    eidx = {e: n for n, e in enumerate(edges)}
    w = np.zeros(len(edges)); d = np.array([Lg[e] for e in edges])
    area_v = defaultdict(float)
    for t in range(len(F)):
        v = F[t]
        a = Lg[key(v[1], v[2])]; b = Lg[key(v[2], v[0])]; c = Lg[key(v[0], v[1])]
        s = 0.5*(a+b+c)
        A = np.sqrt(max(s*(s-a)*(s-b)*(s-c), 0.0))
        for corner, (o, p, q) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
            cot = (p*p + q*q - o*o)/(4*A) if A > 0 else 0.0
            w[eidx[key(v[(corner+1) % 3], v[(corner+2) % 3])]] += 0.5*cot
        for vv in v: area_v[vv] += A/3.0
    return np.array(edges), d, w, area_v
