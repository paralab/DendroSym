"""cascade_tensor_eqn.py -- generic structured looped SIMD emitter.

System-agnostic. A `TensorEqn` describes one cascade layer as an index-
structured equation (symbolic free indices, `Sum(...)` contractions,
`IndexedBase` leaves). `emit_tensor_eqn` turns it into nested C++ `for`-loops
over the free indices with the contraction unrolled as a VEC-macro FMA chain --
the same loop shape the hand oracle produces, but driven from the index
structure so it works for any tensor system. `emit_looped_body` assembles a
complete kernel body from an ordered list of items (TensorEqns + verbatim
irregular-layer lines) supplied by a PDE-system module (bssn_looped, mhd_looped,
...). Nothing here is BSSN-specific.

This is the structured path. It never touches the flat IR pipeline
(bssn_cascade -> cascade_builder), which fully unrolls and blows the AVX2
instruction cache. Leaves stay as VEC arrays in scope (igt[i][l], C1[l][j][k]);
state/derivs are loaded once by a system-supplied prologue; intermediates are
array locals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import sympy as sym

from dendrosym.cascade.vec_printer import to_vec_cpp


# ----------------------------------------------------------------------------
# TensorEqn IR
# ----------------------------------------------------------------------------

@dataclass
class TensorEqn:
    """One cascade layer as an index-structured equation.

    out      : output tensor array name (e.g. "C2" -> `VEC C2[3][3][3]`).
    free     : free (looped) index names, in C++ subscript order ("i","j","k").
    body     : sympy expr over IndexedBase leaves (igt[i,l], C1[l,j,k]) and
               Sum(expr, (l, 0, 2)) contractions; free indices stay symbolic.
    ranges   : per-index extent (default 3 for any index not listed).
    symmetry : symmetric free-index pairs, e.g. (("i","j"),). The pair is
               looped canonically (second index starts at the first) and the
               mirror write is aliased. First member must precede the second
               in `free`. At most one pair is supported.
    """
    out: str
    free: Tuple[str, ...]
    body: sym.Expr
    ranges: Dict[str, int] = field(default_factory=dict)
    symmetry: Tuple[Tuple[str, str], ...] = ()

    def extent(self, idx: str) -> int:
        return self.ranges.get(idx, 3)


# ----------------------------------------------------------------------------
# Array-access naming
# ----------------------------------------------------------------------------

ArrayNamer = Callable[[str, Tuple], str]


def default_array_namer(base: str, idx: Tuple) -> str:
    """Map an IndexedBase access to a C++ multi-dim array access.

    `igt[i,l]` -> "igt[i][l]"; concrete ints pass through ("igt[i][0]")."""
    return base + "".join(f"[{x}]" for x in idx)


def _index_token(ix) -> object:
    """A bound index resolved to a concrete int, or a free index's name str."""
    if ix.is_Integer:
        return int(ix)
    return str(ix)


def _leaf_to_symbol(indexed: sym.Indexed, namer: ArrayNamer) -> sym.Symbol:
    base = indexed.base.label.name if hasattr(indexed.base, "label") else str(indexed.base)
    toks = tuple(_index_token(ix) for ix in indexed.indices)
    return sym.Symbol(namer(base, toks))


# ----------------------------------------------------------------------------
# Body reduction: expand contractions, array-ify leaves, CSE
# ----------------------------------------------------------------------------

@dataclass
class _Reduced:
    temps: list           # [(Symbol, expr), ...] loop-body CSE temps
    final: sym.Expr       # reduced RHS expression (over array-name Symbols)
    full: sym.Expr        # final with temps back-substituted (for verification)


def reduce_body(eqn: TensorEqn, namer: ArrayNamer = default_array_namer) -> _Reduced:
    """Expand Sum contractions to concrete terms, rewrite IndexedBase leaves to
    array-access Symbols, and CSE the resulting body once (over symbolic free
    indices). Returns CSE temps + reduced RHS."""
    body = eqn.body.doit()  # unroll all Sum contractions to concrete bound idx
    subs = {ix: _leaf_to_symbol(ix, namer) for ix in body.atoms(sym.Indexed)}
    body = body.xreplace(subs)

    replacements, reduced = sym.cse([body])
    final = reduced[0]
    temps = [(sym.Symbol(f"{eqn.out}_t{n}"), e) for n, (_, e) in enumerate(replacements)]
    # cse names its temps x0,x1,...; rename to <out>_t<n> to avoid scope clashes.
    rename = {old: new for (new, _), (old, _) in zip(temps, replacements)}
    temps = [(s, e.xreplace(rename)) for s, e in temps]
    final = final.xreplace(rename)

    full = final
    for s, e in reversed(temps):
        full = full.xreplace({s: e})
    return _Reduced(temps=temps, final=final, full=full)


# ----------------------------------------------------------------------------
# Loop emission
# ----------------------------------------------------------------------------

def _sym_partner(idx: str, symmetry) -> Optional[str]:
    """If idx is the *second* member of a symmetric pair, return the first."""
    for a, b in symmetry:
        if idx == b:
            return a
    return None


def emit_tensor_eqn(eqn: TensorEqn, namer: ArrayNamer = default_array_namer,
                    fma: bool = True, declare: bool = True,
                    indent: str = "    ", split: int = 1) -> list:
    """Emit a cascade layer as nested for-loops over the free indices, with the
    contraction unrolled as a VEC-macro FMA chain. Output array is declared
    (unless declare=False) and written canonically; symmetric mirror is aliased.
    A scalar output (no free indices) becomes a single `const VEC` reduction.

    The body uses VEC macros (VFMA/VMUL/...) so the same emit compiles under the
    scalar / AVX2 / AVX-512 macro headers, exactly like the hand oracle.

    split=k>1 breaks each long contraction into k independent FMA accumulator
    chains (pure reassociation) to expose ILP where the serial chain is
    latency-bound -- see to_vec_cpp."""
    if len(eqn.symmetry) > 1:
        raise NotImplementedError("at most one symmetric index pair supported")

    red = reduce_body(eqn, namer)
    rank = len(eqn.free)
    lines: list = []

    # Scalar output (no free indices): a single const VEC reduction, no loops.
    if rank == 0:
        for s, e in red.temps:
            lines.append(f"const VEC {s.name} = {to_vec_cpp(e, fma=fma, split=split)};")
        lines.append(f"const VEC {eqn.out} = {to_vec_cpp(red.final, fma=fma, split=split)};")
        return lines

    if declare:
        dims = "".join(f"[{eqn.extent(ix)}]" for ix in eqn.free)
        lines.append(f"VEC {eqn.out}{dims};")

    depth = 0
    for ix in eqn.free:
        start = _sym_partner(ix, eqn.symmetry)
        lo = start if start is not None else "0"
        lines.append(indent * depth +
                     f"for (int {ix} = {lo}; {ix} < {eqn.extent(ix)}; {ix}++) {{")
        depth += 1

    pad = indent * depth
    for s, e in red.temps:
        lines.append(pad + f"const VEC {s.name} = {to_vec_cpp(e, fma=fma, split=split)};")
    out_acc = namer(eqn.out, tuple(eqn.free))
    lines.append(pad + f"{out_acc} = {to_vec_cpp(red.final, fma=fma, split=split)};")

    # Alias the symmetric mirror (swap the two paired index names in the write).
    for a, b in eqn.symmetry:
        swapped = tuple(a if x == b else (b if x == a else x) for x in eqn.free)
        mirror = namer(eqn.out, swapped)
        lines.append(pad + f"if ({a} != {b}) {mirror} = {out_acc};")

    for d in range(depth - 1, -1, -1):
        lines.append(indent * d + "}")
    return lines


# ----------------------------------------------------------------------------
# Whole-body driver (system-agnostic)
# ----------------------------------------------------------------------------

# SIMD width is orthogonal to the tensor system: (VEC typedef, VFNMADD expansion).
# The fnmadd string is the full replacement text for VFNMADD(a,b,c) = -a*b+c. The
# body is #included inside a function scope, so scalar uses a plain macro
# expression (no helper -- a static function can't be defined there).
_LOOPED_SIMD = {
    "scalar": ("double",  "(-(a)*(b)+(c))"),
    "avx2":   ("__m256d", "_mm256_fnmadd_pd((a),(b),(c))"),
    "avx512": ("__m512d", "_mm512_fnmadd_pd((a),(b),(c))"),
}


def emit_looped_body(items, simd: str = "avx2", banner=None,
                     namer: ArrayNamer = default_array_namer, split: int = 1) -> str:
    """Assemble a complete looped kernel body from an ordered list of `items`.

    Each item is either a `TensorEqn` (emitted as nested loops via
    emit_tensor_eqn) or a `str` of verbatim C++ (comments, and irregular-layer
    lines a system module pulls from its own templates). System-agnostic: the
    caller supplies the items; this driver knows only SIMD width and the
    VEC-macro scaffold. The same body compiles under AVX2/AVX-512 wrappers.

    The tree-FMA printer folds subtractions into VFNMADD, which the shared VEC
    macro headers don't define, so we add it here next to the typedef. Widths:
    scalar (VEC=double) / avx2 / avx512, dispatched via macros_header."""
    from dendrosym.cascade.common import macros_header, indent as _indent
    if simd not in _LOOPED_SIMD:
        raise ValueError(f"emit_looped_body: simd must be one of "
                         f"{tuple(_LOOPED_SIMD)}, got {simd!r}")
    typedef, fnmadd = _LOOPED_SIMD[simd]
    macros = macros_header(simd)

    lines = [macros]
    for b in (banner or []):
        lines.append(f"// {b}")
    lines += [
        "#undef VFNMADD",
        f"#define VFNMADD(a,b,c) {fnmadd}",
        "{",
        f"    typedef {typedef} VEC;  // scoped to this body",
    ]
    body: list = []
    for it in items:
        if isinstance(it, TensorEqn):
            body += emit_tensor_eqn(it, namer=namer, split=split)
        elif isinstance(it, str):
            body.append(it)
        else:
            raise TypeError(f"emit_looped_body item must be TensorEqn or str, "
                           f"got {type(it).__name__}")
    lines += _indent(body)
    lines.append("}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Self-test (generic; toy equations only -- no PDE-system coupling)
# ----------------------------------------------------------------------------

def _selftest() -> None:
    print("=== cascade_tensor_eqn self-test ===")

    # Toy contraction layer: T[i][j][k] = A^{il} B[l][j][k].
    i, j, k, l = sym.symbols("i j k l", integer=True)
    A = sym.IndexedBase("A")
    B = sym.IndexedBase("B")
    eqn = TensorEqn(out="T", free=("i", "j", "k"),
                    body=sym.Sum(A[i, l] * B[l, j, k], (l, 0, 2)))
    red = reduce_body(eqn)
    expected = sum(sym.Symbol(f"A[i][{l}]") * sym.Symbol(f"B[{l}][j][k]")
                   for l in range(3))
    assert sym.expand(red.full - expected) == 0, sym.expand(red.full - expected)
    print("  contraction reduces correctly")

    code = "\n".join(emit_tensor_eqn(eqn))
    assert "VEC T[3][3][3];" in code
    assert "for (int i = 0; i < 3; i++) {" in code
    assert "for (int k = 0; k < 3; k++) {" in code
    assert "T[i][j][k] = " in code
    assert "VFMA(" in code and "VMUL(" in code
    print("  contraction emits a looped FMA chain")

    # Scalar reduction: s = A^{kl} B[k][l] -> single const VEC, no loops.
    seqn = TensorEqn(out="s", free=(),
                     body=sym.Sum(A[k, l] * B[k, l], (k, 0, 2), (l, 0, 2)))
    scode = "\n".join(emit_tensor_eqn(seqn))
    assert scode.startswith("const VEC s = ") and "for (" not in scode
    print("  scalar reduction emits a single const VEC")

    # Symmetry: a symmetric rank-2 layer loops canonically and aliases.
    a, b = sym.symbols("a b", integer=True)
    M = sym.IndexedBase("M")
    v = sym.IndexedBase("v")
    yeqn = TensorEqn(out="S", free=("a", "b"),
                     body=sym.Sum(M[a, l] * v[b, l], (l, 0, 2)),
                     symmetry=(("a", "b"),))
    ycode = "\n".join(emit_tensor_eqn(yeqn))
    assert "for (int b = a; b < 3; b++) {" in ycode
    assert "if (a != b) S[b][a] = S[a][b];" in ycode
    print("  symmetric layer loops canonical + aliases mirror")

    # Driver: a mixed item list (verbatim + TensorEqn) wraps into a SIMD body.
    body = emit_looped_body(["// prologue line", eqn], simd="avx2",
                            banner=["toy looped body"])
    assert "typedef __m256d VEC;" in body and "#define VFNMADD" in body
    assert "// prologue line" in body and "T[i][j][k] = " in body
    print("  emit_looped_body assembles items into a SIMD body")

    print("=== all self-tests passed ===")


if __name__ == "__main__":
    _selftest()
