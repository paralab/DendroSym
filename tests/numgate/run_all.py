#!/usr/bin/env python
"""run_all.py -- self-checks for the numeric differential gate.

    python tests/numgate/run_all.py

These check the gate's own machinery, not any solver: that the DAG evaluator
agrees with sympy on small expressions, that leaf values are reproducible
*across processes* (the property the whole cross-config comparison rests on),
that mixed second derivatives canonicalize to one value, and that the gate
actually reports a disagreement when there is one. Plain asserts, no pytest.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mpmath  # noqa: E402
import sympy as sym  # noqa: E402

from dendrosym import numgate  # noqa: E402

FAILURES = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        FAILURES.append(name)
        print(f"FAIL  {name}\n      {exc}")
    else:
        print(f"PASS  {name}")


def test_numeval_matches_sympy():
    mpmath.mp.dps = 50
    x, y = sym.symbols("x y")
    e = (x + y) ** 3 - 2 * x * y + sym.Rational(3, 7) * x ** 2 / y
    vals = {x: mpmath.mpf("0.3"), y: mpmath.mpf("1.7")}
    got, _ = numgate.numeval(e, vals)
    want = mpmath.mpf(str(e.subs({x: sym.Rational(3, 10), y: sym.Rational(17, 10)})
                          .evalf(50)))
    assert abs(got - want) < mpmath.mpf("1e-45"), f"{got} vs {want}"


def test_numeval_handles_shared_substructure():
    """The memo must not confuse two nodes; a wrong id reuse shows up here."""
    mpmath.mp.dps = 50
    x = sym.Symbol("x")
    sub = (x + 1) ** 2
    e = sub * sub + sub
    vals = {x: mpmath.mpf("0.25")}
    got, nodes = numgate.numeval(e, vals)
    want = mpmath.mpf("1.5625") ** 2 + mpmath.mpf("1.5625")
    assert abs(got - want) < mpmath.mpf("1e-45"), f"{got} vs {want}"
    assert nodes > 0


def test_leaf_values_are_name_deterministic():
    """Two processes must agree on inputs without sharing an RNG stream."""
    a, b = sym.symbols("alpha beta")
    v1 = numgate.leaf_values([a + b], seed=7)
    v2 = numgate.leaf_values([b * b + a], seed=7)   # different tree, same leaves
    assert v1[a] == v2[a] and v1[b] == v2[b], "leaf values depend on traversal order"
    v3 = numgate.leaf_values([a + b], seed=8)
    assert v3[a] != v1[a], "seed has no effect"


def test_mixed_second_derivatives_canonicalize():
    """grad2(2,1,X) and grad2(1,2,X) are one buffer; they must get one value."""
    grad2 = sym.Function("grad2")
    chi = sym.Symbol("chi")
    lo = grad2(1, 2, chi)
    hi = grad2(2, 1, chi)
    assert numgate.canonical_name(lo) == numgate.canonical_name(hi), (
        "mixed partials do not canonicalize -- they would get two values"
    )


def test_unknown_function_is_rejected():
    """An unevaluated node would let two sides agree for the wrong reason."""
    x = sym.Symbol("x")
    e = sym.Function("mystery")(x)
    try:
        numgate.numeval(e, {x: mpmath.mpf(1)})
    except TypeError:
        return
    raise AssertionError("unknown function silently evaluated")


def test_compare_flags_a_difference():
    a = {"u": mpmath.mpf("1.0"), "v": mpmath.mpf("2.0")}
    b = {"u": mpmath.mpf("1.0"), "v": mpmath.mpf("2.0") * (1 + mpmath.mpf("1e-20"))}
    ok, rows = numgate.compare(a, b, tol=mpmath.mpf("1e-40"))
    assert not ok, "identical-looking tables compared equal at 1e-40"
    assert rows[0][0] == "v", f"worst row should be v, got {rows[0][0]}"
    ok2, _ = numgate.compare(a, a, tol=mpmath.mpf("1e-40"))
    assert ok2, "a table disagreed with itself"


def test_dependency_closure_prunes():
    x, y, z = sym.symbols("x y z")
    by_name = {"a": x + 1, "b": sym.Symbol("a") * 2, "c": z ** 2}
    got = numgate.dependency_closure(by_name, ["b"])
    assert got == ["a", "b"], got


if __name__ == "__main__":
    check("numeval matches sympy", test_numeval_matches_sympy)
    check("numeval handles shared substructure", test_numeval_handles_shared_substructure)
    check("leaf values are name-deterministic", test_leaf_values_are_name_deterministic)
    check("mixed second derivatives canonicalize", test_mixed_second_derivatives_canonicalize)
    check("unknown function is rejected", test_unknown_function_is_rejected)
    check("compare flags a difference", test_compare_flags_a_difference)
    check("dependency closure prunes", test_dependency_closure_prunes)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all numgate checks passed")
