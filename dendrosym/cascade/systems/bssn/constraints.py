"""Hamiltonian and momentum constraints as a cascade spec.

The constraints reuse the geometric sector of the BSSN right-hand side (inverse
metric, Christoffels, Ricci) and then assemble two very differently sized
outputs from it: the Hamiltonian constraint is one scalar built from the full
Ricci contraction, the momentum constraint is three components built from
derivatives of At. They are evaluated every step in production, so they serve
as an independent system for the register-pressure predictor of the paper's
null-control figure, at peak live counts between the small control systems and
the full right-hand side.

Expressions transcribed from DendroGR CodeGen/constraints.py.
"""
from collections import OrderedDict

import sympy as sym
from sympy import Rational

from dendrosym.cascade.systems.bssn.legacy import bssn
from dendrosym.cascade.systems.bssn.legacy import dendro
from dendrosym.cascade.systems.bssn.cascade import build_specs
from dendrosym.cascade.systems.bssn.physics import compute_physics


def build_constraint_specs(which="ham,mom"):
    """Declared object sequence ending in the requested constraints.

    which : "ham", "mom", or "ham,mom"
    """
    want = set(which.split(","))
    p = compute_physics()
    chi, K, Gt, At = bssn.chi, bssn.K, bssn.Gt, bssn.At
    igt, d = p.igt, bssn.d

    outs = OrderedDict()
    if "ham" in want:
        outs["ham"] = (sum(chi * igt[j, k] * p.R[j, k] for j, k in dendro.e_ij)
                       - p.At_sqr + Rational(2, 3) * K ** 2)
    if "mom" in want:
        for i in range(3):
            t1 = sum(igt[j, k] * (d(k, At[i, j])
                                  - sum(p.C2[m, k, i] * At[j, m]
                                        for m in dendro.e_i))
                     for j, k in dendro.e_ij)
            t2 = sum(Gt[j] * At[i, j] for j in dendro.e_i)
            t3 = Rational(3, 2) * sum(igt[j, k] * At[k, i] * d(j, chi) / chi
                                      for j, k in dendro.e_ij)
            t4 = Rational(2, 3) * d(i, K)
            outs[f"mom{i}"] = t1 - t2 - t3 - t4

    # the geometric sector, declared exactly as the right-hand side declares it
    geo = [s for s in build_specs()[0] if s[0] != "rhs_assembly"]
    specs = geo + [("constraints", outs)]

    leaves = set()
    for e in outs.values():
        leaves |= e.free_symbols
    for _n, o in geo:
        for e in o.values():
            leaves |= e.free_symbols
    leaves -= {sym.Symbol(k) for _n, o in specs for k in o.keys()}
    return specs, leaves


if __name__ == "__main__":
    import sys
    from dendrosym.cascade.autolayer import (explode, finest_build, auto_chunks,
                                   true_peak)
    for which in ("mom", "ham", "ham,mom"):
        specs, leaves = build_constraint_specs(which)
        objects = explode(specs)
        names = [n for n, _ in objects]
        fin = finest_build(objects, leaves)
        producer, consumes = {}, []
        for i, c in enumerate(fin.chunks):
            for nm in c.outputs:
                producer[nm] = i
            used = set()
            for _, e in c.cse_temps:
                used |= {s.name for s in e.free_symbols}
            for e in c.outputs.values():
                used |= {s.name for s in e.free_symbols}
            consumes.append(used)
        targets = {n for n in names if n.startswith(("ham", "mom"))}
        need = {i for i, n in enumerate(names) if n in targets}
        changed = True
        while changed:
            changed = False
            for i in sorted(need):
                for s_ in consumes[i]:
                    j = producer.get(s_)
                    if j is not None and j not in need:
                        need.add(j); changed = True
        kept = [objects[i] for i in sorted(need)]
        chunks = auto_chunks(kept, leaves, L=None, work_aware=True)
        peak, temps, nl = true_peak(chunks, leaves)
        print(f"{which:9s} objects {len(kept):3d}  L {nl:2d}  peak {peak:4d}  "
              f"temps {temps:4d}", flush=True)
