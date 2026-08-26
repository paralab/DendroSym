#!/usr/bin/env python3
"""Order-optimality check: does ANY valid evaluation order achieve a lower
bottleneck carry than the declared (blackboard) order?

Exact search over linear extensions of the object DAG via a DP on ideals
(downward-closed subsets): B(S) = the best achievable value of
max-over-prefixes of W, over all orders that realize S as a prefix, where
W(S) = sum of widths of placed objects with a consumer still unplaced.
The carry convention matches cascade_autolayer.dependency_carry (terminal
outputs do not carry). Objective is the order-dependent part of the layering
model; cut placement is downstream and unaffected by this check.

Usage: python cascade_order_check.py [--system bssn|emda] [--max-ideals N]
"""
import argparse
from collections import OrderedDict

from dendrosym.cascade.autolayer import explode, finest_build


def build_consumers(objects, fin):
    n = len(objects)
    name_to_idx = {}
    for i, c in enumerate(fin.chunks):
        for nm in c.outputs.keys():
            name_to_idx[nm] = i
    cons = [set() for _ in range(n)]     # cons[i] = objects consuming i's outputs
    preds = [set() for _ in range(n)]    # preds[j] = objects j depends on
    for j, c in enumerate(fin.chunks):
        used = set()
        for _, e in c.cse_temps:
            used |= {s.name for s in e.free_symbols}
        for e in c.outputs.values():
            used |= {s.name for s in e.free_symbols}
        for nm in used:
            i = name_to_idx.get(nm)
            if i is not None and i != j:
                cons[i].add(j)
                preds[j].add(i)
    return cons, preds


def optimal_order(objects, fin, max_ideals=5_000_000):
    """Exact bottleneck-optimal linear extension of the object DAG.
    Returns a list of object indices, or None if the ideal count explodes."""
    cons, preds = build_consumers(objects, fin)
    n = len(objects)
    width = [len(c.outputs) for c in fin.chunks]
    consmask = [sum(1 << j for j in cons[i]) for i in range(n)]
    predmask = [sum(1 << j for j in preds[i]) for i in range(n)]
    FULL = (1 << n) - 1

    def W(S):
        w = 0
        for i in range(n):
            if (S >> i) & 1 and consmask[i] & ~S:
                w += width[i]
        return w

    B = {0: 0}
    parent = {0: (None, None)}
    frontier = {0: 0}
    for _ in range(n):
        nxt = {}
        for S, b in frontier.items():
            for v in range(n):
                bit = 1 << v
                if S & bit or (predmask[v] & S) != predmask[v]:
                    continue
                S2 = S | bit
                b2 = max(b, W(S2))
                if b2 < nxt.get(S2, 1 << 30) and b2 < B.get(S2, 1 << 30):
                    nxt[S2] = b2
                    parent[S2] = (S, v)
        B.update(nxt)
        frontier = nxt
        if len(B) > max_ideals:
            return None
    order = []
    S = FULL
    while S:
        S0, v = parent[S]
        order.append(v)
        S = S0
    order.reverse()
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=("bssn", "emda"), default="bssn")
    ap.add_argument("--max-ideals", type=int, default=5_000_000)
    args = ap.parse_args()

    if args.system == "emda":
        from dendrosym.cascade.systems.emda.cascade import build_emda_specs
        specs, leaves = build_emda_specs()
    else:
        from dendrosym.cascade.systems.bssn.cascade import build_specs
        specs, leaves = build_specs()

    objects = explode(specs)
    fin = finest_build(objects, leaves)
    cons, preds = build_consumers(objects, fin)
    n = len(objects)
    width = [len(c.outputs) for c in fin.chunks]
    consmask = [sum(1 << j for j in cons[i]) for i in range(n)]
    predmask = [sum(1 << j for j in preds[i]) for i in range(n)]
    FULL = (1 << n) - 1

    def W(S):
        w = 0
        for i in range(n):
            if (S >> i) & 1 and consmask[i] & ~S:
                w += width[i]
        return w

    # declared-order bottleneck
    S = 0
    declared = 0
    for i in range(n):
        S |= 1 << i
        declared = max(declared, W(S))

    # DP over ideals
    B = {0: 0}
    parent = {0: (None, None)}
    frontier = {0: 0}
    for _ in range(n):
        nxt = {}
        for S, b in frontier.items():
            for v in range(n):
                bit = 1 << v
                if S & bit or (predmask[v] & S) != predmask[v]:
                    continue
                S2 = S | bit
                b2 = max(b, W(S2))
                if b2 < nxt.get(S2, 1 << 30) and b2 < B.get(S2, 1 << 30):
                    nxt[S2] = b2
                    parent[S2] = (S, v)
        B.update(nxt)
        frontier = nxt
        if len(B) > args.max_ideals:
            print(f"ABORT: ideal count exceeded {args.max_ideals}")
            return
    best = B[FULL]

    # recover one optimal order
    order = []
    S = FULL
    while S:
        S0, v = parent[S]
        order.append(v)
        S = S0
    order.reverse()

    print(f"system {args.system}: {n} objects, {len(B):,} ideals explored")
    print(f"declared-order bottleneck carry : {declared}")
    print(f"optimal bottleneck carry        : {best}")
    if best == declared:
        print("VERDICT: the declared (blackboard) order is carry-optimal.")
    else:
        print(f"VERDICT: a better order exists ({best} vs {declared}):")
        print("  " + " -> ".join(objects[v][0] for v in order))


if __name__ == "__main__":
    main()
