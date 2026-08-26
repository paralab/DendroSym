"""cascade_split.py -- IR-to-IR split transform (Algorithm 3 from talk).

Partition the heaviest chunk's outputs into two sub-chunks until depth reaches
a target L > natural_L. The talk argues that splitting *preserves* sparsity:
the (A, B) coefficient tensors are simply partitioned by row, so no
composition cost is paid and peak-live drops. We mimic that at the spec level
by partitioning the OrderedDict of outputs into bins of roughly-equal cost.

Within a single chunk, BSSN outputs are independent (a_rhs doesn't reference
chi_rhs in the rhs_assembly chunk; ricci's R, Rt, Rphi don't reference each
other; derived_quantities' members likewise). Splitting is therefore a
straight partition with no within-chunk dependency rewiring needed.

Cost heuristic
--------------
Greedy bin-pack by expression-tree size: largest output goes into the
smallest bin. Tends to produce balanced halves.
"""

from collections import OrderedDict

import sympy as sym


def _expr_size(expr) -> int:
    return sum(1 for _ in sym.preorder_traversal(expr))


def _chunk_size(spec) -> int:
    _, outputs = spec
    return sum(_expr_size(e) for e in outputs.values())


def _split_chunk(spec, num_groups: int):
    """Partition a chunk's outputs into num_groups OrderedDicts of roughly
    equal cost. Greedy LPT bin-pack."""
    name, outputs = spec
    items = list(outputs.items())
    if len(items) < num_groups:
        # Trivial: one output per bin, pad with empties (callers should avoid)
        bins_dict = [OrderedDict([items[i]] if i < len(items) else []) for i in range(num_groups)]
        return [(f"{name}_p{i}", b) for i, b in enumerate(bins_dict)]

    sizes = [(_expr_size(e), i) for i, (_, e) in enumerate(items)]
    sizes.sort(reverse=True)  # largest first
    bins = [[] for _ in range(num_groups)]
    bin_sizes = [0] * num_groups
    for sz, idx in sizes:
        b = bin_sizes.index(min(bin_sizes))
        bins[b].append(items[idx])
        bin_sizes[b] += sz
    # Preserve original output ordering within each bin
    out = []
    for i, items_in_bin in enumerate(bins):
        # Stable-sort by original index to keep names like a_rhs, b_rhs0, b_rhs1
        # together when possible. Not required for correctness but makes the
        # generated body easier to read.
        items_in_bin.sort(key=lambda kv: list(outputs.keys()).index(kv[0]))
        out.append((f"{name}_p{i}", OrderedDict(items_in_bin)))
    return out


def split_to_target(specs, target_L: int, verbose: bool = False):
    """Greedy split to reach target_L > len(specs)."""
    if target_L <= len(specs):
        return list(specs)

    specs = list(specs)
    while len(specs) < target_L:
        sizes = [_chunk_size(s) for s in specs]
        biggest = sizes.index(max(sizes))
        # Don't split chunks that are already trivial
        if len(specs[biggest][1]) < 2:
            raise RuntimeError(
                f"cannot split further: target_L={target_L} but largest chunk "
                f"has {len(specs[biggest][1])} outputs"
            )
        sub = _split_chunk(specs[biggest], 2)
        specs = specs[:biggest] + sub + specs[biggest + 1:]
        if verbose:
            print(
                f"  split: chunk {biggest} ({sub[0][0].rsplit('_p', 1)[0]}, "
                f"size={sizes[biggest]}) -> 2; len -> {len(specs)}"
            )
    return specs


# ---------------------------------------------------------------------------
# Smart split (post-CSE, talk's Algorithm 3 proper)
# ---------------------------------------------------------------------------
# Operates on a CascadeResult (post-build). Each smart split:
#   1. Picks one chunk and runs LPT bin-pack on its outputs.
#   2. Walks the chunk's CSE-temp DAG to find which temps each partition
#      transitively depends on.
#   3. Classifies temps: shared (in 2+ partitions) vs private (in exactly 1).
#   4. Replaces the original chunk with [shared_chunk, p0_chunk, p1_chunk].
#      The shared sub-chunk emits the shared temps as named outputs so the
#      following partition sub-chunks can reference them by Symbol name.
#
# This produces 2 extra chunks per split (was 1, now 3). Numerics are
# bit-identical to the original natural-depth cascade because the math is
# unchanged — only the emit order and naming change.

def _transitive_temps(expr, temp_expr_map):
    """Walk an expression's free symbols; for any that's a CSE temp, recurse
    on its RHS. Return the set of all CSE temp names this expression
    transitively depends on."""
    result = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        try:
            free = e.free_symbols
        except AttributeError:
            continue
        for s in free:
            n = s.name
            if n in temp_expr_map and n not in result:
                result.add(n)
                stack.append(temp_expr_map[n])
    return result


def smart_split_chunk(chunk, num_groups: int = 2):
    """Smart split a single ChunkResult into [shared, p0, p1, ...] sub-chunks.

    Parameters
    ----------
    chunk : ChunkResult
        Already CSE'd; `chunk.cse_temps` and `chunk.outputs` are populated.
    num_groups : int
        Number of partitions (2 = halves, 3 = thirds, etc.).

    Returns
    -------
    list of ChunkResult
        Either [p0, p1, ...] (if no shared temps were detected) or
        [shared, p0, p1, ...].
    """
    from collections import OrderedDict
    from dendrosym.cascade.builder import ChunkResult

    items = list(chunk.outputs.items())
    if len(items) < num_groups:
        return [chunk]
    sizes = [_expr_size(e) for _, e in items]

    # LPT bin-pack: largest first into the currently-smallest bin.
    bins = [[] for _ in range(num_groups)]
    bin_sizes = [0] * num_groups
    for idx in sorted(range(len(items)), key=lambda i: -sizes[i]):
        b = bin_sizes.index(min(bin_sizes))
        bins[b].append(idx)
        bin_sizes[b] += sizes[idx]
    bins = [sorted(b) for b in bins]
    output_groups = [OrderedDict([items[i] for i in b]) for b in bins]

    # Build temp lookup; preserve original order for emission.
    temp_expr_map = {s.name: e for s, e in chunk.cse_temps}
    temp_order = [s.name for s, _ in chunk.cse_temps]

    # For each group, collect transitive temp dependencies.
    group_deps = []
    for g in output_groups:
        deps = set()
        for _, e in g.items():
            deps.update(_transitive_temps(e, temp_expr_map))
        group_deps.append(deps)

    # Shared temps: in 2+ groups.
    shared = set()
    for i in range(len(group_deps)):
        for j in range(i + 1, len(group_deps)):
            shared |= group_deps[i] & group_deps[j]

    private_per_group = [d - shared for d in group_deps]

    new_chunks = []
    if shared:
        # Shared temps as outputs of a precursor sub-chunk. Emit in original
        # CSE order so downstream chunks see them after they're declared.
        shared_outputs = OrderedDict(
            (n, temp_expr_map[n]) for n in temp_order if n in shared
        )
        new_chunks.append(ChunkResult(
            name=f"{chunk.name}_shared",
            cse_temps=[],
            outputs=shared_outputs,
            input_symbols=set(),
            n_temps=0,
        ))

    import sympy as sym
    for i, (g_outputs, g_priv) in enumerate(zip(output_groups, private_per_group)):
        # Private temps stay as cse_temps within the half-chunk; emit_cpp_unrolled
        # will lay them out before the half's outputs in order.
        priv_temps = [(sym.Symbol(n), temp_expr_map[n])
                      for n in temp_order if n in g_priv]
        new_chunks.append(ChunkResult(
            name=f"{chunk.name}_p{i}",
            cse_temps=priv_temps,
            outputs=g_outputs,
            input_symbols=set(),
            n_temps=len(priv_temps),
        ))

    return new_chunks


def _chunk_total_size(chunk) -> int:
    """Sum of expression-tree sizes for all outputs (matches the heuristic
    used by the pre-CSE split_to_target so smart and dumb pick the same
    chunks for an apples-to-apples comparison)."""
    return sum(_expr_size(e) for _, e in chunk.outputs.items())


def smart_split_result(result, num_splits: int = 1, verbose: bool = False):
    """Apply num_splits smart splits to a CascadeResult, picking the chunk
    with the largest total expression-tree size each time. Same heuristic
    as the pre-CSE `split_to_target` so smart-split and dumb-split target
    the same chunks for A/B comparison.

    Each smart split adds 2 chunks (shared + 2 halves replace 1 original).
    """
    from dendrosym.cascade.builder import CascadeResult

    chunks = list(result.chunks)
    for k in range(num_splits):
        # Skip chunks already produced by an earlier smart split.
        candidates = [(i, c) for i, c in enumerate(chunks)
                      if not c.name.endswith("_shared")
                      and "_p" not in c.name.rsplit("_", 1)[-1]
                      and len(c.outputs) >= 2]
        if not candidates:
            if verbose:
                print(f"  smart_split: no candidates left at step {k}")
            break
        idx, chosen = max(candidates, key=lambda ic: _chunk_total_size(ic[1]))
        sub = smart_split_chunk(chosen, num_groups=2)
        if verbose:
            n_shared = (len(sub[0].outputs) if sub[0].name.endswith("_shared")
                        else 0)
            print(
                f"  smart_split #{k+1}: split chunk {idx} ({chosen.name}, "
                f"{chosen.n_temps} CSE temps, "
                f"size={_chunk_total_size(chosen)}) -> "
                f"{len(sub)} sub-chunks (shared={n_shared}, "
                f"p0={sub[-2].n_temps if len(sub)>=2 else '?'} priv-temps, "
                f"p1={sub[-1].n_temps} priv-temps)"
            )
        chunks = chunks[:idx] + sub + chunks[idx + 1:]

    return CascadeResult(chunks=chunks, leaf_symbols=result.leaf_symbols)
