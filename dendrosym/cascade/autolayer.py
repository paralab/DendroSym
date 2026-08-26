#!/usr/bin/env python3
"""Automatic layer selection for the polynomial cascade.

Given a spec's physics evaluation ORDER (the object sequence a relativist
writes at the blackboard), choose the LAYER GROUPING automatically:

  1. explode the declared chunks to one-object-per-chunk granularity
     (objects = maximal name-prefix groups of outputs);
  2. build that finest cascade once to obtain the true dependency sets;
  3. boundary carry W(b) at every cut position b is partition-independent
     (an output crosses cut b iff it is produced before b and consumed at
     or after b), so candidate layerings come from a small exact DP over
     cut positions minimizing max over chunks of
         W(entry) + sum of member finest-CSE temps    (the liveness proxy);
  4. every candidate is validated EXACTLY: built via build_cascade_ir,
     emitted unrolled, and measured with the paper's source-liveness parser.

Usage:
    python cascade_autolayer.py            # sweep L = 1..12, report table
    python cascade_autolayer.py --L 7      # best grouping at exactly L=7
"""
import argparse
import hashlib
import os
import re
import sys
import tempfile
from collections import OrderedDict

import sympy as sym


from dendrosym.cascade.builder import CascadeBuilder, build_cascade_ir

PAPER_FIGS = os.environ.get("CASCADE_PAPER_FIGS", os.path.expanduser(
    "~/research/papers/cascade_paper/data/figures"))


def object_prefix(name):
    """Maximal object prefix: strip trailing index digits (C1_012 -> C1_,
    igt00 -> igt, a_rhs -> a_rhs, gt_rhs00 -> gt_rhs)."""
    return re.sub(r"\d+$", "", name)


def explode(specs):
    """Declared chunks -> one chunk per named object, order preserved."""
    objects = []
    for cname, outputs in specs:
        groups = OrderedDict()
        for oname, expr in outputs.items():
            groups.setdefault(object_prefix(oname), OrderedDict())[oname] = expr
        for prefix, outs in groups.items():
            objects.append((prefix.rstrip("_") or prefix, outs))
    return objects


def finest_build(objects, leaves):
    b = CascadeBuilder()
    b.set_leaves(leaves)
    for name, outs in objects:
        b.add_chunk(name, outs)
    return b.build(cse_prefix="AUTO_")


def dependency_carry(objects, result):
    """W[b] for every cut position b (between object b-1 and b), from the
    finest build's post-substitution expressions."""
    n = len(objects)
    out_names = [set(c.outputs.keys()) for c in result.chunks]
    name_to_idx = {}
    for i, names in enumerate(out_names):
        for nm in names:
            name_to_idx[nm] = i
    # last consumer index of each producing object
    last_use = [i for i in range(n)]
    for j, c in enumerate(result.chunks):
        used = set()
        for _, e in c.cse_temps:
            used |= {s.name for s in e.free_symbols}
        for e in c.outputs.values():
            used |= {s.name for s in e.free_symbols}
        for nm in used:
            if nm in name_to_idx and name_to_idx[nm] < j:
                i = name_to_idx[nm]
                last_use[i] = max(last_use[i], j)
    width = [len(names) for names in out_names]
    W = [0] * (n + 1)
    for b in range(1, n):
        W[b] = sum(width[i] for i in range(b) if last_use[i] >= b)
    return W, width, last_use


def dp_partition(objects, W, temps, L):
    """Exact DP: partition [0..n) into L contiguous chunks minimizing
    max over chunks of (W[start] + sum of member finest temps)."""
    n = len(objects)
    INF = float("inf")

    def cost(i, j):     # chunk spanning objects [i, j)
        return (W[i] if i > 0 else 0) + sum(temps[k] for k in range(i, j))

    best = [[INF] * (L + 1) for _ in range(n + 1)]
    cut = [[None] * (L + 1) for _ in range(n + 1)]
    best[0][0] = 0
    for j in range(1, n + 1):
        for l in range(1, min(L, j) + 1):
            for i in range(l - 1, j):
                v = max(best[i][l - 1], cost(i, j))
                if v < best[j][l]:
                    best[j][l] = v
                    cut[j][l] = i
    cuts = []
    j, l = n, L
    while l > 0:
        i = cut[j][l]
        cuts.append(i)
        j, l = i, l - 1
    return sorted(cuts)[1:]  # drop the leading 0



# ---------------------------------------------------------------------------
# Recomputation-aware objective
#
# The liveness-only objective above counts, for a candidate chunk, the sum of
# its members' finest-CSE temporaries.  That over-counts a merged chunk (per-
# chunk CSE collapses subexpressions the members shared) and, worse, it never
# charges a SPLIT for the work it duplicates: a boundary drawn through a tensor
# family makes both sides recompute what they used to share.  On EMDA that is
# visible as the total emitted temporaries rising from 733 to 1{,}262 at deep L
# while the modelled peak falls.
#
# canonical_temp_keys() hash-conses every finest-build temporary so that "the
# same subexpression" is decidable across objects.  With those keys the exact
# emitted-temporary count of any contiguous merge is just the size of the union,
# which makes both the peak proxy and the work count exact functions of the cut
# positions, and the two can be traded against each other on an exact frontier.
# ---------------------------------------------------------------------------

def canonical_temp_keys(result):
    """Per-chunk lists of canonical keys for that chunk's CSE temporaries.

    Two temporaries carry the same key iff they are the same subexpression, so
    the number of distinct keys over a contiguous range of objects is the
    number of temporaries per-chunk CSE emits once that range is merged.
    Hash-consed bottom-up, so no expression is ever expanded.
    """
    per_chunk = []
    for c in result.chunks:
        local = {}
        keys = []
        for symb, expr in c.cse_temps:
            repl = {sym.Symbol(nm): sym.Symbol("#" + k) for nm, k in local.items()}
            e = expr.xreplace(repl) if repl else expr
            k = hashlib.sha1(sym.srepr(e).encode("utf-8")).hexdigest()[:16]
            local[symb.name] = k
            keys.append(k)
        per_chunk.append(keys)
    return per_chunk


def merged_temp_table(keys):
    """U[i][j] = temporaries emitted by the merge of objects [i, j)."""
    n = len(keys)
    U = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        seen = set()
        for j in range(i, n):
            seen.update(keys[j])
            U[i][j + 1] = len(seen)
    return U


def dp_min_work(W, U, L, cap):
    """Partition [0..n) into exactly L chunks minimising total emitted
    temporaries, subject to every chunk's peak proxy W[start] + U[start,end]
    staying at or below cap.  Returns (work, cuts) or None if infeasible."""
    n = len(U) - 1
    INF = float("inf")
    best = [[INF] * (L + 1) for _ in range(n + 1)]
    cut = [[None] * (L + 1) for _ in range(n + 1)]
    best[0][0] = 0
    for j in range(1, n + 1):
        for l in range(1, min(L, j) + 1):
            for i in range(l - 1, j):
                if best[i][l - 1] == INF:
                    continue
                u = U[i][j]
                if (W[i] if i > 0 else 0) + u > cap:
                    continue
                v = best[i][l - 1] + u
                if v < best[j][l]:
                    best[j][l] = v
                    cut[j][l] = i
    if best[n][L] == INF:
        return None
    cuts, j, l = [], n, L
    while l > 0:
        i = cut[j][l]
        cuts.append(i)
        j, l = i, l - 1
    return best[n][L], sorted(cuts)[1:]


def pareto_frontier(W, U, L):
    """Exact (peak proxy, emitted temporaries) frontier at depth L."""
    n = len(U) - 1
    caps = sorted({(W[i] if i > 0 else 0) + U[i][j]
                   for i in range(n) for j in range(i + 1, n + 1)})
    front, seen_work = [], set()
    for cap in caps:
        r = dp_min_work(W, U, L, cap)
        if r is None:
            continue
        work, cuts = r
        peak = max((W[a] if a > 0 else 0) + U[a][b]
                   for a, b in zip([0] + cuts, cuts + [n]))
        if (peak, work) in seen_work:
            continue
        seen_work.add((peak, work))
        front.append((peak, work, cuts))
    # keep only non-dominated points
    out = []
    for peak, work, cuts in sorted(front):
        if out and out[-1][1] <= work:
            continue
        out.append((peak, work, cuts))
    return out


def work_aware_cuts(W, U, L, peak_slack=0.0):
    """Cuts at depth L minimising emitted temporaries subject to the peak
    proxy staying within peak_slack of the best achievable at that depth."""
    front = pareto_frontier(W, U, L)
    if not front:
        return None
    pmin = front[0][0]
    cap = pmin * (1.0 + peak_slack)
    best = None
    for peak, work, cuts in front:
        if peak <= cap and (best is None or work < best[1]):
            best = (peak, work, cuts)
    return best


def merge(objects, cuts):
    bounds = [0] + list(cuts) + [len(objects)]
    chunks = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        outs = OrderedDict()
        for name, o in objects[a:b]:
            outs.update(o)
        label = "_".join(name for name, _ in objects[a:b])
        label = __import__("re").sub(r"[^A-Za-z0-9_]", "_", label)[:40]
        chunks.append((label, outs))
    return chunks


def true_peak(chunks, leaves):
    """Build + emit + measure with the paper's parser. Ground truth."""
    res = build_cascade_ir(chunks, leaves, cse_prefix="AUTO_")
    code = res.emit_cpp_unrolled(dendro_var_style=True, inline_threshold=0,
                                 short_names=False)
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(code)
        path = f.name
    from dendrosym.cascade.liveness_viz import parse_kernel, source_liveness
    stmts, roots, labels = parse_kernel(path)
    lv = source_liveness(stmts)
    os.unlink(path)
    total_temps = sum(c.n_temps for c in res.chunks)
    return lv["peak"], total_temps, len(res.chunks)


def auto_chunks(specs, leaves, L=None, verbose=False, search_order=False,
                work_aware=False, peak_slack=0.0):
    """The one-call API: declared spec -> automatically layered chunk list.

    specs : list[(name, OrderedDict[str, expr])] in evaluation order; the
        grouping is discarded, only the order matters. Reference earlier
        outputs BY SYMBOL (by-value trees break under Add-flattening and
        silently smear the carry model).
    L : target layer count, or None to choose L by the DP objective
        (argmin over 1..n_objects of the optimal minimax chunk cost).
    work_aware : charge a candidate grouping for the temporaries a split
        duplicates, and break ties on that count instead of accepting any
        boundary that shaves the modelled peak.  Off by default so the
        published BSSN kernels regenerate byte for byte.
    peak_slack : in work-aware mode, how far above the best achievable peak
        proxy a grouping may sit in exchange for less recomputation.
    Returns the merged chunk list, ready for build_cascade_ir().
    """
    objects = explode(specs)
    fin = finest_build(objects, leaves)
    if search_order:
        from dendrosym.cascade.order_check import optimal_order
        idx = optimal_order(objects, fin)
        if idx is not None:
            objects = [objects[i] for i in idx]
            fin = finest_build(objects, leaves)
            if verbose:
                print("auto_chunks: order searched ->",
                      " ".join(name for name, _ in objects))
    W, _, _ = dependency_carry(objects, fin)
    temps = [c.n_temps for c in fin.chunks]

    if work_aware:
        U = merged_temp_table(canonical_temp_keys(fin))
        n_obj = len(objects)
        if L is None:
            best = None
            for l in range(1, n_obj + 1):
                r = work_aware_cuts(W, U, l, peak_slack)
                if r is None:
                    continue
                peak, work, cuts = r
                if best is None or (peak, work) < (best[0], best[1]):
                    best = (peak, work, l, cuts)
            peak, work, L, cuts = best
        else:
            peak, work, cuts = work_aware_cuts(W, U, L, peak_slack)
        if verbose:
            print(f"auto_chunks: work-aware L={L} peak proxy {peak}, "
                  f"{work} emitted temps")
            print("auto_chunks: cuts before "
                  f"{[objects[c][0] for c in cuts]}")
        return merge(objects, cuts)

    def objective(cuts):
        bounds = [0] + list(cuts) + [len(objects)]
        return max((W[a] if a > 0 else 0) + sum(temps[a:b])
                   for a, b in zip(bounds[:-1], bounds[1:]))

    if L is None:
        best = None
        for l in range(1, len(objects) + 1):
            cuts = dp_partition(objects, W, temps, l)
            score = objective(cuts)
            if best is None or score < best[0]:
                best = (score, l, cuts)
        _, L, cuts = best
        if verbose:
            print(f"auto_chunks: chose L={L} (objective {best[0]})")
    else:
        cuts = dp_partition(objects, W, temps, L)
    if verbose:
        print(f"auto_chunks: cuts before {[objects[c][0] for c in cuts]}")
    return merge(objects, cuts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--ssl", action="store_true")
    ap.add_argument("--cahd", action="store_true")
    ap.add_argument("--system", choices=("bssn", "emda"), default="bssn")
    ap.add_argument("--work-aware", action="store_true")
    ap.add_argument("--peak-slack", type=float, default=0.0)
    ap.add_argument("--frontier", action="store_true")
    args = ap.parse_args()

    if args.system == "emda":
        from dendrosym.cascade.systems.emda.cascade import build_emda_specs
        specs, leaves = build_emda_specs()
    else:
        from dendrosym.cascade.systems.bssn.cascade import build_specs
        specs, leaves = build_specs(ssl=args.ssl, cahd=args.cahd)
    objects = explode(specs)
    print(f"exploded {len(specs)} declared chunks -> {len(objects)} objects:")
    print("  " + " | ".join(name for name, _ in objects))

    fin = finest_build(objects, leaves)
    W, width, last_use = dependency_carry(objects, fin)
    temps = [c.n_temps for c in fin.chunks]
    print("\ncut-position carry W(b) (partition-independent):")
    for b in range(1, len(objects)):
        print(f"  before {objects[b][0]:22s} W={W[b]:3d}")

    U = merged_temp_table(canonical_temp_keys(fin))

    if args.frontier:
        L = args.L or 7
        print(f"\nexact (peak proxy, emitted temps) frontier at L={L}:")
        for peak, work, cuts in pareto_frontier(W, U, L):
            names = [objects[c][0] for c in cuts]
            print(f"  proxy {peak:4d}  temps {work:5d}  cuts before: {names}")
        return

    Ls = [args.L] if args.L else range(1, 13)
    print(f"\n{'L':>3} {'TRUE peak':>9} {'temps':>6}  boundaries")
    for L in Ls:
        if L > len(objects):
            break
        if args.work_aware:
            r = work_aware_cuts(W, U, L, args.peak_slack)
            if r is None:
                continue
            cuts = r[2]
        else:
            cuts = dp_partition(objects, W, temps, L)
        chunks = merge(objects, cuts)
        peak, ntemps, ln = true_peak(chunks, leaves)
        names = [objects[c][0] for c in cuts]
        print(f"{L:3d} {peak:9d} {ntemps:6d}  cuts before: {names}")


if __name__ == "__main__":
    main()
