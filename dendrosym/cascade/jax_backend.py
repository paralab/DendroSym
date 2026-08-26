"""cascade_jax.py -- generic JAX source emitter for the structured cascade.

System-agnostic. Consumes the same `TensorEqn` specs as the C++ looped emitter
(cascade_tensor_eqn) and emits a standalone, vectorized JAX module: each layer
becomes a `jnp.einsum` (or broadcast) over whole tensors. Nothing here is
BSSN-specific -- a PDE-system module supplies the `TensorEqn` list (the same one
it feeds to emit_looped_body), and this turns it into importable JAX source.

Where the C++ path UNROLLS contractions into FMA chains (narrow SIMD wants flat
instruction streams), the JAX path KEEPS them as contractions -- `Sum_l A[i,l]
B[l,j]` -> `einsum('il,lj->ij', A, B)`.

PERFORMANCE NOTE (measured, RTX 3060): for BSSN this einsum form is CORRECT
(fp64-gated 24/24 @ 2.8e-16 vs emit_jax_kernel) but SLOWER than the flat
lambdify path -- ~15-19x slower in fp64, because BSSN tensors are tiny+fixed
(3x3, 3x3x3): many small einsum kernels lose to XLA fusing the flat scalar
arithmetic into a few big elementwise kernels. Use the flat emit_jax_kernel for
fast BSSN-on-GPU; this einsum emitter is the readable/general path (and may pay
off for systems with larger tensors). Mirror-image of the C++ result, where the
same structure HELPS (AVX2 icache).

Leaves and intermediates are jnp arrays whose leading axes are the tensor indices
(shape (3,)*rank); an outer vmap over grid points batches them. Output array axis
order follows `TensorEqn.free`.
"""

from __future__ import annotations

from typing import List, Tuple

import sympy as sym

from dendrosym.cascade.tensor_eqn import TensorEqn


# ----------------------------------------------------------------------------
# Coefficient / factor helpers
# ----------------------------------------------------------------------------

def _fmt_coeff(c: sym.Expr) -> str:
    """Format a sympy Number as an exact Python float literal."""
    if c.is_Integer:
        return f"{int(c)}.0"
    if c.is_Rational:
        return f"({int(c.p)}.0/{int(c.q)}.0)"
    return repr(float(c))


def _strip_sums(expr: sym.Expr) -> sym.Expr:
    """Drop every `Sum(...)` wrapper, keeping the (symbolic-index) summand.

    einsum infers the contracted indices from "present in operands, absent from
    the output", so the explicit Sum is redundant once we know the free set."""
    return expr.replace(lambda e: isinstance(e, sym.Sum), lambda e: e.function)


def _decompose_term(term: sym.Expr):
    """Split one additive monomial into (coeff, indexed_factors, scalar_factors).

    indexed_factors: list of (base_name, [index letters]) -- einsum operands
    (a Pow of an Indexed is expanded to repeated operands).
    scalar_factors: list of str -- bare Symbol leaves (0-d, broadcast)."""
    coeff, rest = term.as_coeff_Mul()
    indexed: List[Tuple[str, List[str]]] = []
    scalars: List[str] = []
    for f in sym.Mul.make_args(rest):
        if isinstance(f, sym.Indexed):
            indexed.append((f.base.name, [str(ix) for ix in f.indices]))
        elif isinstance(f, sym.Pow) and isinstance(f.base, sym.Indexed) \
                and f.exp.is_Integer and f.exp > 0:
            for _ in range(int(f.exp)):
                indexed.append((f.base.base.name, [str(ix) for ix in f.base.indices]))
        elif isinstance(f, sym.Symbol):
            scalars.append(f.name)
        elif isinstance(f, sym.Pow) and isinstance(f.base, sym.Symbol) \
                and f.exp.is_Integer:
            scalars.append(f"{f.base.name}**{int(f.exp)}")
        else:
            raise NotImplementedError(f"cascade_jax: unsupported factor {f!r} "
                                      f"in term {term!r}")
    return coeff, indexed, scalars


def _term_to_jax(term: sym.Expr, free: Tuple[str, ...]) -> Tuple[bool, str]:
    """Emit one additive monomial as a jnp expression. Returns (is_negative,
    code) so the caller can join terms with +/-."""
    coeff, indexed, scalars = _decompose_term(term)
    neg = coeff.is_negative
    if neg:
        coeff = -coeff

    parts: List[str] = []
    if coeff != 1:
        parts.append(_fmt_coeff(coeff))
    parts.extend(scalars)
    if indexed:
        spec = ",".join("".join(idx) for _, idx in indexed) + "->" + "".join(free)
        ops = ", ".join(base for base, _ in indexed)
        parts.append(f"jnp.einsum('{spec}', {ops})")
    if not parts:                      # bare +/-1 constant
        parts.append("1.0")
    return neg, " * ".join(parts)


# ----------------------------------------------------------------------------
# Single-equation and whole-module emission
# ----------------------------------------------------------------------------

def tensor_eqn_to_jax(eqn: TensorEqn) -> str:
    """Right-hand side of one TensorEqn as a jnp expression (einsum per term)."""
    body = sym.expand(_strip_sums(eqn.body))
    pieces: List[str] = []
    for i, term in enumerate(sym.Add.make_args(body)):
        neg, code = _term_to_jax(term, eqn.free)
        if i == 0:
            pieces.append(("-" + code) if neg else code)
        else:
            pieces.append((" - " if neg else " + ") + code)
    return "".join(pieces) if pieces else "0.0"


def _referenced_bases(eqns: List[TensorEqn]) -> List[str]:
    """All IndexedBase names + scalar-Symbol leaves referenced across the eqn
    bodies. Excludes the tensor-index symbols (i,j,k,...), which are einsum
    subscripts, not array inputs."""
    index_syms = set()
    for eqn in eqns:
        for a in eqn.body.atoms(sym.Indexed):
            index_syms.update(str(ix) for ix in a.indices if ix.is_Symbol)
    names = set()
    for eqn in eqns:
        for a in eqn.body.atoms(sym.Indexed):
            names.add(a.base.name)
        for s in eqn.body.atoms(sym.Symbol):
            if s.name not in index_syms:
                names.add(s.name)
    return sorted(names)


def emit_jax_module(eqns: List[TensorEqn], fn_name: str = "cascade_rhs",
                    docstring: str = "") -> str:
    """Emit a standalone JAX module: a jitted function `fn_name(leaves)` that
    takes a dict of input arrays, computes each TensorEqn in order (outputs of
    earlier layers are visible to later ones), and returns a dict of outputs.

    Generic: the eqn list is the only input. Leaf inputs are auto-detected as
    every referenced name that isn't produced by an earlier equation."""
    produced: List[str] = []
    body_lines: List[str] = []
    seen = set()
    for eqn in eqns:
        body_lines.append(f"    {eqn.out} = {tensor_eqn_to_jax(eqn)}")
        if eqn.out not in seen:
            produced.append(eqn.out)
            seen.add(eqn.out)
    produced_set = set(produced)
    leaves = [n for n in _referenced_bases(eqns) if n not in produced_set]

    out = [
        '"""', docstring or f"Auto-generated JAX cascade ({fn_name}).",
        "Generated by cascade_jax.emit_jax_module from TensorEqn specs.",
        "Vectorized: each layer is a jnp.einsum over tensor axes; vmap over points.",
        '"""',
        "import jax",
        "import jax.numpy as jnp",
        "",
        "",
        "@jax.jit",
        f"def {fn_name}(leaves):",
        f"    # leaves: dict with keys {leaves}",
    ]
    for n in leaves:
        out.append(f"    {n} = leaves['{n}']")
    out.append("")
    out += body_lines
    out.append("")
    ret = ", ".join(f"'{n}': {n}" for n in produced)
    out.append(f"    return {{{ret}}}")
    out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Self-test (generic; toy equations; numeric check via numpy.einsum)
# ----------------------------------------------------------------------------

def _selftest() -> None:
    import numpy as np
    print("=== cascade_jax self-test ===")

    i, j, k, l, m = sym.symbols("i j k l m", integer=True)
    A = sym.IndexedBase("A")
    B = sym.IndexedBase("B")
    v = sym.IndexedBase("v")
    s = sym.Symbol("s")

    # 1. single contraction -> clean einsum
    e1 = TensorEqn(out="C", free=("i", "j", "k"),
                   body=sym.Sum(A[i, l] * B[l, j, k], (l, 0, 2)))
    code1 = tensor_eqn_to_jax(e1)
    assert code1 == "jnp.einsum('il,ljk->ijk', A, B)", code1
    print("  contraction -> einsum:", code1)

    # 2. double contraction to scalar
    e2 = TensorEqn(out="t", free=(),
                   body=sym.Sum(A[k, l] * A[k, l], (k, 0, 2), (l, 0, 2)))
    code2 = tensor_eqn_to_jax(e2)
    assert code2 == "jnp.einsum('kl,kl->', A, A)", code2
    print("  scalar reduction ->", code2)

    # 3. sum of terms with scalar broadcast + outer product + coefficient
    e3 = TensorEqn(out="R", free=("i", "j"),
                   body=s * A[i, j] - sym.Rational(1, 4) * v[i] * v[j])
    code3 = tensor_eqn_to_jax(e3)
    print("  mixed term ->", code3)

    # numeric check (numpy stands in for jnp): build the einsum, compare to a
    # brute-force loop over the original (Sum-expanded) expression.
    rng = np.random.default_rng(0)
    Av = rng.standard_normal((3, 3)); Bv = rng.standard_normal((3, 3, 3))
    got = np.einsum('il,ljk->ijk', Av, Bv)
    want = np.zeros((3, 3, 3))
    for ii in range(3):
        for jj in range(3):
            for kk in range(3):
                want[ii, jj, kk] = sum(Av[ii, ll] * Bv[ll, jj, kk] for ll in range(3))
    assert np.allclose(got, want), "einsum != brute-force contraction"
    print("  numeric einsum matches brute-force contraction")

    # 4. whole-module emission + chaining (C2 then CalGt-style reduction)
    mod = emit_jax_module([e1], fn_name="toy_rhs")
    assert "@jax.jit" in mod and "jnp.einsum('il,ljk->ijk', A, B)" in mod
    assert "leaves['A']" in mod and "leaves['B']" in mod and "'C': C" in mod
    print("  emit_jax_module: ok")

    print("=== all self-tests passed ===")


if __name__ == "__main__":
    _selftest()
