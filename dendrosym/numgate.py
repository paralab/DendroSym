"""Numeric differential gate: do two derivations of the same physics agree?

The pipeline's normal gate is *bit-identity* of the emitted gencode, which is
exact and cheap -- but it only applies when a change is meant to be a no-op. A
config refactor that legitimately restructures the algebra (index-notation
rewrites, a symmetric-matrix helper, staging, a different CSE grouping) produces
different expression trees for the same physics, and bit-identity cannot score
it. Today that class of change ships with no gate at all.

This module evaluates both forms numerically, at the same random field values,
and reports the relative disagreement per output. It is what found the nine
`rPerpPerpPerpPerp` entries that had been left as array-initialiser integers:
two derivations of the Cotton tensor disagreed, and nothing else would have said
so.

What it catches: a refactor that changed the physics.
What it does NOT catch: physics that was already wrong before the refactor --
both forms derive from the same config, so both are wrong together. That class
needs a reference, not a tool.

Three pieces do the work:

* :func:`numeval` -- a DAG-memoised mpmath evaluator. ``subs()``/``evalf()`` do
  not finish on multi-million-op expressions; walking the shared DAG once does.
* :func:`dependency_closure` -- evaluate only what the targets actually need.
* :func:`leaf_values` -- leaf values derived from the leaf's *canonical name*,
  not from a shared RNG stream, so two configs evaluated in **separate
  processes** still agree on inputs. Separate processes are mandatory: configs
  call ``dendrosym.nr.set_metric`` at import, so two of them cannot coexist.
"""

import hashlib

import mpmath
import sympy as sym

# Derivative markers are dynamically-created grad/grad2/... FUNCTION
# applications, not nr.d/nr.d2s classes. `expr.atoms(SomeClass)` silently
# returns an empty set for them; match atoms(Function) by type name instead.
MARKERS = ("grad", "grad2", "agrad", "kograd")

# sympy functions that may legitimately appear in a config's algebra. Anything
# not listed raises rather than being silently skipped -- an unevaluated node
# would make two sides agree for the wrong reason.
DEFAULT_FUNCS = {
    "log": mpmath.log,
    "exp": mpmath.exp,
    "sqrt": mpmath.sqrt,
    "sin": mpmath.sin,
    "cos": mpmath.cos,
    "tan": mpmath.tan,
    "atan": mpmath.atan,
    "sinh": mpmath.sinh,
    "cosh": mpmath.cosh,
    "tanh": mpmath.tanh,
    "Abs": abs,
}


def is_marker(atom):
    return atom.is_Function and type(atom).__name__ in MARKERS


def markers(expr):
    """Every derivative marker appearing in `expr`."""
    return {a for a in expr.atoms(sym.Function) if is_marker(a)}


def other_functions(expr):
    """Names of non-marker function atoms -- the dispatch table must cover these."""
    return {type(a).__name__ for a in expr.atoms(sym.Function) if not is_marker(a)}


def canonical_name(atom):
    """A name for a leaf that two independent derivations will both produce.

    Derivative markers are canonicalized by SORTING their index arguments:
    ``grad2(2,1,chi)`` and ``grad2(1,2,chi)`` are the same mixed partial and the
    generated code stores them in one buffer, so they must receive one value.
    Giving them two is a silent disagreement that looks like a physics error --
    it cost a debugging cycle the first time.
    """
    if atom.is_Symbol:
        return f"sym:{atom.name}"
    if is_marker(atom):
        args = list(atom.args)
        idx = sorted(str(a) for a in args[:-1])
        return f"{type(atom).__name__}:{'.'.join(idx)}:{args[-1]}"
    raise TypeError(f"not a leaf: {atom!r}")


def _uniform_from_name(name, seed, lo, hi):
    """Deterministic uniform draw keyed by name, not by draw order.

    A shared `random.seed` stream only reproduces if both sides request leaves in
    the same order, which two different expression trees do not.
    """
    h = hashlib.sha256(f"{seed}|{name}".encode()).digest()
    frac = int.from_bytes(h[:8], "big") / float(1 << 64)
    return mpmath.mpf(lo) + (mpmath.mpf(hi) - mpmath.mpf(lo)) * mpmath.mpf(frac)


_DIAGONAL_SUFFIXES = ("00", "11", "22")


def _base_name(leaf):
    """Symbol name without its array subscript.

    Leaves reach us already indexed -- ``gt00[pp]``, not ``gt00`` -- because the
    config carries the memory access. Classifying on the raw string silently
    matches nothing, which is how every metric component ended up drawn small and
    ``det(gt)`` came out negative.
    """
    name = str(leaf)
    return name.split("[", 1)[0]


def leaf_values(exprs, seed=0, positive_prefixes=("chi", "alpha", "psi", "W")):
    """Assign a value to every free symbol and derivative marker in `exprs`.

    The draw has to produce a state that is non-degenerate but still *valid*, and
    those pull in opposite directions:

    * `positive_prefixes` names fields sitting under fractional powers
      (``chi**1.5`` and friends); drawn positive and O(1).
    * Rank-2 components whose name ends 00/11/22 are drawn near 1 and the
      off-diagonals small, so a 3-metric built from them stays positive-definite.
      Drawing all six uniformly about zero makes ``det`` negative about half the
      time, and the first ``det**(1/3)`` then returns a COMPLEX number -- which
      surfaces as a crash, or worse as two sides "agreeing" on garbage.
    * Derivative markers are drawn small: they are gradients of O(1) fields.

    NOTE: this deliberately picks a non-degenerate state. A check run where the
    quantities all vanish -- Minkowski, for a curvature block -- tests the
    plumbing, not the labelling, and cannot detect a permutation of zeros.
    """
    leaves = set()
    for e in exprs:
        leaves |= e.free_symbols
        leaves |= markers(e)

    vals = {}
    for leaf in sorted(leaves, key=lambda a: canonical_name(a)):
        name = canonical_name(leaf)
        if is_marker(leaf):
            vals[leaf] = _uniform_from_name(name, seed, -0.15, 0.15)
            continue
        base = _base_name(leaf)
        if any(base.startswith(p) for p in positive_prefixes):
            vals[leaf] = _uniform_from_name(name, seed, 0.6, 1.4)
        elif base.endswith(_DIAGONAL_SUFFIXES):
            vals[leaf] = _uniform_from_name(name, seed, 0.85, 1.15)
        elif len(base) > 2 and base[-2:].isdigit():
            vals[leaf] = _uniform_from_name(name, seed, -0.12, 0.12)
        else:
            vals[leaf] = _uniform_from_name(name, seed, -0.3, 0.3)
    return vals


def numeval(expr, vals, funcs=None):
    """Evaluate `expr` at `vals`, memoised on the shared DAG.

    Returns ``(value, n_nodes)``. `vals` maps leaf atoms (symbols and derivative
    markers) to mpmath numbers.

    Keyed on ``id(node)``, so the cache stores ``(node, result)``: the node
    reference keeps Python from freeing the object and reusing its id for a
    different node. Dropping it corrupts results silently.
    """
    funcs = DEFAULT_FUNCS if funcs is None else funcs
    cache = {}

    def go(x):
        k = id(x)
        hit = cache.get(k)
        if hit is not None:
            return hit[1]
        if x.is_Symbol or is_marker(x):
            try:
                r = vals[x]
            except KeyError:
                raise KeyError(f"no value supplied for leaf {x}") from None
        elif x.is_Number:
            if x.is_Rational:
                r = mpmath.mpf(int(x.p)) / int(x.q)
            elif x.is_Float:
                r = mpmath.mpf(float(x))
            else:
                r = mpmath.mpf(str(x))
        elif x.is_Add:
            r = mpmath.fsum([go(a) for a in x.args])
        elif x.is_Mul:
            r = mpmath.mpf(1)
            for a in x.args:
                r *= go(a)
        elif x.is_Pow:
            r = go(x.args[0]) ** go(x.args[1])
        elif x.is_Function and type(x).__name__ in funcs:
            r = funcs[type(x).__name__](*[go(a) for a in x.args])
        else:
            raise TypeError(f"cannot evaluate {type(x).__name__}: {x}")
        cache[k] = (x, r)
        return r

    return go(expr), len(cache)


def dependency_closure(by_name, targets):
    """Names in `by_name` that `targets` transitively need, in `by_name` order.

    Evaluating a whole staged block to check three of its outputs is the
    difference between seconds and never finishing.
    """
    want, stack = set(), list(targets)
    while stack:
        n = stack.pop()
        if n in want or n not in by_name:
            continue
        want.add(n)
        stack.extend(str(x) for x in by_name[n].free_symbols if str(x) in by_name)
    return [n for n in by_name if n in want]


def evaluate_block(names, exprs, staged_names=(), staged_exprs=(), seed=0,
                   targets=None, dps=60, funcs=None):
    """Evaluate a config's outputs at deterministic leaf values.

    Staged quantities are resolved first, in dependency order, and their values
    fed in as leaves for the outputs that reference them by symbol.

    Returns ``{output_name: mpmath value}``.
    """
    mpmath.mp.dps = dps
    staged_by_name = dict(zip([str(n) for n in staged_names], staged_exprs))

    wanted = list(names) if targets is None else [n for n in names if n in targets]
    out_exprs = [e for n, e in zip(names, exprs) if n in set(wanted)]

    needed = dependency_closure(
        staged_by_name,
        [str(s) for e in out_exprs for s in e.free_symbols
         if str(s) in staged_by_name],
    )
    staged_used = [staged_by_name[n] for n in needed]

    unhandled = set()
    for e in out_exprs + staged_used:
        unhandled |= other_functions(e) - set(
            (DEFAULT_FUNCS if funcs is None else funcs)
        )
    if unhandled:
        raise TypeError(
            f"unhandled function atoms {sorted(unhandled)}; extend the dispatch "
            "table rather than letting them evaluate to something arbitrary"
        )

    vals = leaf_values(out_exprs + staged_used, seed=seed)
    for n in needed:
        vals[sym.Symbol(n)] = numeval(staged_by_name[n], vals, funcs)[0]

    out = {n: numeval(e, vals, funcs)[0] for n, e in zip(wanted, out_exprs)}

    # A complex result means the random state was not physically admissible --
    # log() or a fractional power hit a negative argument, typically because the
    # conformal metric drawn was not positive-definite. Comparing two complex
    # numbers would still "agree", so this must fail loudly rather than pass.
    bad = sorted(n for n, v in out.items() if isinstance(v, mpmath.mpc))
    if bad:
        raise ValueError(
            f"complex value(s) for {bad[:4]}{' ...' if len(bad) > 4 else ''}: the "
            "random field state is not admissible (negative determinant or "
            "negative log argument). Adjust leaf_values / positive_prefixes for "
            "this config rather than comparing complex results."
        )
    return out


def compare(a, b, tol=mpmath.mpf("1e-40")):
    """Compare two ``{name: value}`` tables; return ``(ok, rows)``.

    `rows` is ``(name, value_a, value_b, relative_difference)`` sorted worst
    first, so a disagreement points at which quantity to look at rather than
    just saying no.
    """
    names = sorted(set(a) | set(b))
    rows, ok = [], True
    for n in names:
        if n not in a or n not in b:
            rows.append((n, a.get(n), b.get(n), mpmath.inf))
            ok = False
            continue
        va, vb = a[n], b[n]
        # a complex value means some base went negative under a fractional
        # power: report it rather than crashing, since the magnitude comparison
        # is still meaningful and the imaginary part is the diagnostic.
        scale = max(abs(va), abs(vb), mpmath.mpf(1))
        rel = abs(va - vb) / scale
        if rel > tol:
            ok = False
        rows.append((n, va, vb, rel))
    rows.sort(key=lambda r: -r[3])
    return ok, rows


def format_report(rows, limit=None):
    out = []
    for n, va, vb, rel in rows[: limit or len(rows)]:
        out.append(f"  {n:<28} rel {mpmath.nstr(rel, 6):>14}")
    return "\n".join(out)
