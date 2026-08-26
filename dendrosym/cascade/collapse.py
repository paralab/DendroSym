"""cascade_collapse.py -- IR-to-IR collapse transform (Algorithm 2 from talk).

Greedy merge of adjacent chunks in a CascadeBuilder spec list until the depth
matches a target L < natural_L. Operates on pre-CSE specs (the
`(name, OrderedDict[output_name -> sym.Expr])` tuples produced by
`bssn_cascade.build_specs()`), so the merged result re-runs CSE at build()
time and the emitter sees a well-formed cascade with fewer chunks.

Cost heuristic
--------------
We pick the boundary whose merge has the *smallest combined expression-tree
size*. That's a cheap proxy for "post-merge CSE blowup": small chunks merge
first, deep heavy chunks like the rhs_assembly stay separated longer.

The talk's slide 16 says "collapsing destroys sparsity by composing two
quadratics into a wide quadratic." Merging chunk specs simulates that: the
resulting chunk contains both layers' outputs, and per-chunk CSE has to find
shared subexpressions across both inside one larger pool — exactly the
register-pressure regime we're studying.
"""

from collections import OrderedDict

import sympy as sym


def _expr_size(expr) -> int:
    """Approximate cost: number of nodes in the SymPy expression tree."""
    return sum(1 for _ in sym.preorder_traversal(expr))


def _chunk_size(spec) -> int:
    _, outputs = spec
    return sum(_expr_size(e) for e in outputs.values())


def _merge(spec_a, spec_b):
    """Concatenate two chunk specs into one, substituting spec_a's outputs
    into spec_b's expressions so the merged chunk has no within-chunk Symbol
    references.

    The substitution is the talk's "compose two quadratics" semantics: after
    merging, B_{ℓ,ℓ+1} is the composed coefficient tensor over the merged
    input set. Per-chunk CSE inside the merged chunk recovers any shared
    sub-products as anonymous temps; the named outputs from the upper chunk
    are still emitted (so downstream chunks that reference them keep working)
    but they're no longer the sole evaluation order — they're recomputed
    if cheaper, and re-aliased back to their named symbols as outputs.
    """
    a_subs = {sym.Symbol(name): expr for name, expr in spec_a[1].items()}
    b_subbed = OrderedDict(
        (name, expr.xreplace(a_subs)) for name, expr in spec_b[1].items()
    )
    merged = OrderedDict()
    merged.update(spec_a[1])
    merged.update(b_subbed)
    # Use `_x_` (not `+`) so the chunk name remains a valid C++ identifier
    # when CascadeBuilder forms its CSE prefix `CASC_<NAME>_`.
    name = f"{spec_a[0]}_x_{spec_b[0]}"
    return (name, merged)


def collapse_to_target(specs, target_L: int, verbose: bool = False):
    """Greedy collapse to reach target_L < len(specs).

    Returns a new list of specs of length target_L.
    """
    if target_L >= len(specs):
        return list(specs)
    if target_L < 1:
        raise ValueError(f"target_L must be >= 1, got {target_L}")

    specs = list(specs)
    while len(specs) > target_L:
        # Score each adjacent pair by combined size; pick the smallest.
        sizes = [_chunk_size(s) for s in specs]
        best_i, best_cost = None, None
        for i in range(len(specs) - 1):
            cost = sizes[i] + sizes[i + 1]
            if best_cost is None or cost < best_cost:
                best_i, best_cost = i, cost
        merged = _merge(specs[best_i], specs[best_i + 1])
        specs = specs[:best_i] + [merged] + specs[best_i + 2:]
        if verbose:
            print(
                f"  collapse: merged at boundary {best_i} "
                f"(cost={best_cost}); len -> {len(specs)} -> {merged[0]}"
            )
    return specs
