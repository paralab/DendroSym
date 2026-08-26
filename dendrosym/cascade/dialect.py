"""cascade_dialect.py -- naming-dialect abstraction for cascade drivers.

Different PDE systems use different component-naming conventions for the same
underlying tensors:

    BSSN harness        EMDA / emda-gr
    ------------        --------------
    gt0..gt5            gt00..gt22
    At0..At5            At00..At22
    K                   trK
    B0..B2              gaugeB0..gaugeB2
    Gt0..Gt2            CAP_Gt0..CAP_Gt2
    a                   alpha            (lapse)
    eta                 etadamp          (BSSN gauge damping)

cascade_builder.add_chunk() takes raw string keys (e.g. {"gt00": expr}), so
the naming dialect is the *caller's* responsibility. This module centralises
it so a PDE driver picks a dialect and the chunk-output keys come out right.

`TensorSpec` is the same shape as in `legacy/cascade_tensor.py`, deliberately:
when the tensor-pattern detector lands, the explicit-driver pipeline (here)
and the auto-detector pipeline can speak the same vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

import sympy as sym


# ----------------------------------------------------------------------------
# TensorSpec + factories  (matches legacy/cascade_tensor.py)
# ----------------------------------------------------------------------------

@dataclass
class TensorSpec:
    """How a logical tensor lays out as flat symbols.

    components maps every index tuple (e.g. (0,1), (1,0) for sym tensors) to
    the same component string ("gt01"). Symmetric tensors store both orders;
    callers don't need to canonicalise indices before lookup.
    """
    name: str
    rank: int
    symmetric: bool = False
    components: Dict[Tuple[int, ...], str] = field(default_factory=dict)

    def component(self, *idx: int) -> str:
        return self.components[idx]


def make_scalar(name: str, component: Optional[str] = None) -> TensorSpec:
    """Rank-0. `component` defaults to `name`."""
    return TensorSpec(
        name=name, rank=0, symmetric=False,
        components={(): component or name},
    )


def make_vec3(name: str, prefix: Optional[str] = None) -> TensorSpec:
    """Rank-1: name0, name1, name2.  Use `prefix` to override the leaf name."""
    p = prefix if prefix is not None else name
    return TensorSpec(
        name=name, rank=1, symmetric=False,
        components={(i,): f"{p}{i}" for i in range(3)},
    )


def make_sym33_2digit(name: str, prefix: Optional[str] = None) -> TensorSpec:
    """Sym 3x3 with two-digit naming (gt00..gt22). EMDA / emda-gr convention."""
    p = prefix if prefix is not None else name
    components: Dict[Tuple[int, int], str] = {}
    for i in range(3):
        for j in range(3):
            a, b = sorted((i, j))
            components[(i, j)] = f"{p}{a}{b}"
    return TensorSpec(name=name, rank=2, symmetric=True, components=components)


def make_sym33_flat(name: str, prefix: Optional[str] = None) -> TensorSpec:
    """Sym 3x3 with flat-6 naming (gt0..gt5). Harness BSSN convention."""
    p = prefix if prefix is not None else name
    flat = {(0, 0): 0, (0, 1): 1, (0, 2): 2,
            (1, 1): 3, (1, 2): 4, (2, 2): 5}
    components: Dict[Tuple[int, int], str] = {}
    for i in range(3):
        for j in range(3):
            ij = tuple(sorted((i, j)))
            components[(i, j)] = f"{p}{flat[ij]}"
    return TensorSpec(name=name, rank=2, symmetric=True, components=components)


def make_deriv_family(name: str, base: "TensorSpec", order: int = 1,
                      array: Optional[str] = None) -> TensorSpec:
    """Derivative tensor family for the structured looped emitter.

    The d-th (and, for order 2, e-th) spatial derivative of a base tensor adds
    leading derivative indices: order-1 -> (d, *base_idx); order-2 ->
    (d, e, *base_idx). Components map to *array-access* strings (d_gt[d][i][j]),
    not flat symbols -- the looped emitter consumes the leaf name verbatim as a
    VEC array access. Symmetry of the base tensor is preserved on its indices.

    Generic naming: array stem defaults to "d_<name>" (order 1) / "d2_<name>"
    (order 2); override with `array` for system-specific stems (e.g. "d_al").
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    stem = array if array is not None else (("d_" if order == 1 else "d2_") + name)
    components: Dict[Tuple[int, ...], str] = {}
    deriv_ranges = [range(3)] * order
    import itertools
    base_idx = sorted(base.components.keys())
    for dtuple in itertools.product(*deriv_ranges):
        for bidx in base_idx:
            full = (*dtuple, *bidx)
            acc = "".join(f"[{x}]" for x in full)
            components[full] = f"{stem}{acc}"
    return TensorSpec(name=stem, rank=order + base.rank,
                      symmetric=base.symmetric, components=components)


def make_tensor3(name: str, prefix: Optional[str] = None,
                 sym_last2: bool = False) -> TensorSpec:
    """Rank-3 3x3x3 (Christoffel-like). `prefix` overrides the leaf stem
    (e.g. "C2_" -> C2_012). sym_last2=True maps (j,k) and (k,j) to the same
    sorted name (18 distinct); otherwise all 27 are distinct."""
    p = prefix if prefix is not None else name
    components: Dict[Tuple[int, int, int], str] = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if sym_last2:
                    a, b = sorted((j, k))
                    components[(i, j, k)] = f"{p}{i}{a}{b}"
                else:
                    components[(i, j, k)] = f"{p}{i}{j}{k}"
    return TensorSpec(name=name, rank=3, symmetric=sym_last2, components=components)


# ----------------------------------------------------------------------------
# NamingDialect
# ----------------------------------------------------------------------------

@dataclass
class NamingDialect:
    """A bundle of TensorSpecs keyed by logical tensor name.

    Drivers call `dialect.spec(logical_name)` to get the naming for the output
    keys, and `dialect.rename_map()` to get a flat str->str mapping suitable
    for `xreplace` over symbolic expressions (when the same tensor appears in
    different dialects in upstream code).
    """
    name: str
    tensors: Dict[str, TensorSpec] = field(default_factory=dict)

    def add(self, spec: TensorSpec) -> None:
        self.tensors[spec.name] = spec

    def spec(self, name: str) -> TensorSpec:
        return self.tensors[name]

    def has(self, name: str) -> bool:
        return name in self.tensors


# ----------------------------------------------------------------------------
# Dialect-aware flatten helpers  (mirror bssn_cascade.flatten_*  signatures)
# ----------------------------------------------------------------------------

def flatten_scalar(expr: sym.Expr, name: str, dialect: NamingDialect) -> "OrderedDict[str, sym.Expr]":
    from collections import OrderedDict
    spec = dialect.spec(name)
    return OrderedDict([(spec.component(), expr)])


def flatten_vec3(vec, name: str, dialect: NamingDialect) -> "OrderedDict[str, sym.Expr]":
    """vec is a length-3 indexable (list, sym.Matrix, or 1-d array)."""
    from collections import OrderedDict
    spec = dialect.spec(name)
    out = OrderedDict()
    for i in range(3):
        out[spec.component(i)] = vec[i]
    return out


def flatten_sym33(mat, name: str, dialect: NamingDialect) -> "OrderedDict[str, sym.Expr]":
    """mat is a 3x3 sym.Matrix (or anything indexable as mat[i, j])."""
    from collections import OrderedDict
    spec = dialect.spec(name)
    out = OrderedDict()
    seen = set()
    for i in range(3):
        for j in range(3):
            key = spec.component(i, j)
            if key in seen:
                continue
            seen.add(key)
            out[key] = mat[i, j]
    return out


def flatten_tensor3(tensor, name: str, dialect: NamingDialect) -> "OrderedDict[str, sym.Expr]":
    """3x3x3 with last-two-indices symmetry (Christoffel-like).

    `dialect.spec(name)` must define components for all 27 (i,j,k) tuples
    or just 18 if you're treating (j,k) as symmetric. For generality we just
    look up each index tuple.
    """
    from collections import OrderedDict
    spec = dialect.spec(name)
    out = OrderedDict()
    seen = set()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                try:
                    key = spec.component(i, j, k)
                except KeyError:
                    # Fall back to default Christoffel-style naming if the
                    # dialect doesn't carry an explicit rank-3 spec.
                    a, b = sorted((j, k))
                    key = f"{name}_{i}{a}{b}"
                if key in seen:
                    continue
                seen.add(key)
                out[key] = tensor[i, j, k]
    return out


# ----------------------------------------------------------------------------
# Symbol renaming between dialects
# ----------------------------------------------------------------------------

def rename_symbols(expr: sym.Expr, renames: Mapping[str, str]) -> sym.Expr:
    """Rewrite all `sym.Symbol(old)` -> `sym.Symbol(new)` in `expr`."""
    subs = {sym.Symbol(old): sym.Symbol(new) for old, new in renames.items()}
    return expr.xreplace(subs)


def rename_chunk(outputs: Mapping[str, sym.Expr],
                 key_renames: Mapping[str, str],
                 symbol_renames: Mapping[str, str]) -> "OrderedDict[str, sym.Expr]":
    """Apply key and symbol renaming uniformly. Either map may be empty."""
    from collections import OrderedDict
    subs = {sym.Symbol(o): sym.Symbol(n) for o, n in symbol_renames.items()}
    out = OrderedDict()
    for k, v in outputs.items():
        k2 = key_renames.get(k, k)
        v2 = v.xreplace(subs) if subs else v
        out[k2] = v2
    return out


# ----------------------------------------------------------------------------
# Standard dialects
# ----------------------------------------------------------------------------

def _bssn_dialect_2digit(include_rank3: bool = False) -> NamingDialect:
    """BSSN with two-digit component names. This matches what bssn_cascade.py
    emits in its `gt_rhs00..gt_rhs22` outputs (NOT the harness flat form).

    `include_rank3` (default OFF) registers the rank-3 Christoffels C1/C2/C3.
    GATED OFF: nothing consumes these yet; they are kept as a verified reference
    for the future structured looped emitter. Opt in explicitly to use them."""
    d = NamingDialect(name="bssn_2digit")
    for t in ["gt", "igt", "At", "At_UU", "AikAkj", "DiDj_a", "tf", "R"]:
        d.add(make_sym33_2digit(t))
    for t in ["Gt", "CalGt", "b", "B"]:
        d.add(make_vec3(t))
    for t in ["a", "chi", "K", "chi_inv", "At_sqr", "lap_a", "eta"]:
        d.add(make_scalar(t))
    if include_rank3:
        # Rank-3 Christoffels (underscore naming, C1 symmetric in last two).
        d.add(make_tensor3("C1", prefix="C1_", sym_last2=True))
        d.add(make_tensor3("C2", prefix="C2_", sym_last2=False))
        d.add(make_tensor3("C3", prefix="C3_", sym_last2=False))
    # RHS outputs use the *_rhs suffix; encode as scalars/vecs that wrap the
    # underlying tensor's spec.
    d.add(make_scalar("a_rhs"))
    d.add(make_scalar("chi_rhs"))
    d.add(make_scalar("K_rhs"))
    d.add(make_vec3("b_rhs"))
    d.add(make_vec3("Gt_rhs"))
    d.add(make_vec3("B_rhs"))
    d.add(make_sym33_2digit("gt_rhs"))
    d.add(make_sym33_2digit("At_rhs"))
    return d


def _emda_dialect() -> NamingDialect:
    """EMDA naming (emda-gr / harness_emda).

    Differences vs BSSN 2-digit:
        a        -> alpha
        K        -> trK
        Gt[i]    -> CAP_Gt[i]
        B[i]     -> gaugeB[i]
        eta      -> etadamp     (BSSN gauge damping; EMDA reserves `eta[2]`
                                 for matter sector)
    Matter sector adds: dilatonPhi, kappa, capitalPi, capitalXi,
                        perpE[0..2], perpB[0..2], dampingPsi, dampingPhi.
    """
    d = NamingDialect(name="emda")
    # Vacuum BSSN tensors -- same 2-digit shape but different prefixes for
    # the vector / scalar slots.
    for t in ["gt", "igt", "At", "At_UU", "AikAkj", "DiDj_a", "tf", "R"]:
        d.add(make_sym33_2digit(t))
    for t in ["CalGt", "b"]:
        d.add(make_vec3(t))
    d.add(make_vec3("Gt", prefix="CAP_Gt"))
    d.add(make_vec3("B", prefix="gaugeB"))
    d.add(make_scalar("a", component="alpha"))
    d.add(make_scalar("K", component="trK"))
    d.add(make_scalar("eta", component="etadamp"))
    for t in ["chi", "chi_inv", "At_sqr", "lap_a"]:
        d.add(make_scalar(t))
    # RHS outputs follow the EMDA dialect for the LHS:
    d.add(make_scalar("a_rhs", component="alpha_rhs"))
    d.add(make_scalar("chi_rhs"))
    d.add(make_scalar("K_rhs", component="trK_rhs"))
    d.add(make_vec3("b_rhs", prefix="beta_rhs"))
    d.add(make_vec3("Gt_rhs", prefix="CAP_Gt_rhs"))
    d.add(make_vec3("B_rhs", prefix="gaugeB_rhs"))
    d.add(make_sym33_2digit("gt_rhs"))
    d.add(make_sym33_2digit("At_rhs"))
    # Matter state variables (scalar/vec) -- prefer the harness-canonical names.
    for s in ["dilatonPhi", "kappa", "capitalPi", "capitalXi",
              "dampingPsi", "dampingPhi"]:
        d.add(make_scalar(s))
        d.add(make_scalar(f"{s}_rhs"))
    for v in ["perpE", "perpB"]:
        d.add(make_vec3(v))
        d.add(make_vec3(f"{v}_rhs"))
    return d


BSSN_DIALECT = _bssn_dialect_2digit()
EMDA_DIALECT = _emda_dialect()


# Sym33 index map: flat-6 (BSSN harness) -> 2-digit (EMDA).
_SYM6_TO_2DIGIT = {0: "00", 1: "01", 2: "02", 3: "11", 4: "12", 5: "22"}

# (regex pattern, replacement). Order matters; longer patterns first.
# Same logic as rename_cascade_to_emda.RENAMES but used at the Symbol-name
# level (not the emitted-text level).
_BSSN_TO_EMDA_PATTERNS: list[Tuple[str, str]] = []


def _build_patterns() -> None:
    import re  # noqa: F401  -- used by translate_bssn_to_emda_name
    P = _BSSN_TO_EMDA_PATTERNS

    # Symbol renames. Each tensor gets TWO patterns:
    #   deriv-context: matches X[i] when preceded by `_` (grad_d_X..., agrad_..)
    #   plain-context: matches X[i] when preceded by non-identifier
    # Deriv-context MUST run before plain-context, otherwise after `Gt0` ->
    # `CAP_Gt0` the deriv pattern re-matches the `_Gt0` substring inside
    # the rename target and produces `CAP_CAP_Gt0`. (Gt is the only target
    # whose rename string itself contains `_<name><idx>` as a substring,
    # but the ordering is the safe convention regardless.)

    # --- deriv-context (run first) ---
    for i, ij in _SYM6_TO_2DIGIT.items():
        for kind in ("gt", "At"):
            P.append((rf"(?<=_){kind}{i}(?![0-9])", f"{kind}{ij}"))
    P.append((r"(?<=_)K(?![A-Za-z0-9_])", "trK"))
    for c in range(3):
        P.append((rf"(?<=_)B{c}(?![A-Za-z0-9_])",  f"gaugeB{c}"))
        P.append((rf"(?<=_)Gt{c}(?![A-Za-z0-9_])", f"CAP_Gt{c}"))
        P.append((rf"(?<=_)b{c}(?![A-Za-z0-9_])",  f"beta{c}"))

    # --- plain-context (run after) ---
    for i, ij in _SYM6_TO_2DIGIT.items():
        for kind in ("gt", "At"):
            P.append((rf"(?<![A-Za-z0-9_]){kind}{i}(?![0-9])", f"{kind}{ij}"))

    # RHS-output key suffix swap: gt_rhs00 -> gt00_rhs (and At_rhs00 -> At00_rhs).
    for ij in ("00", "01", "02", "11", "12", "22"):
        P.append((rf"\bgt_rhs{ij}\b", f"gt{ij}_rhs"))
        P.append((rf"\bAt_rhs{ij}\b", f"At{ij}_rhs"))

    # K -> trK  (K_rhs first so the longer match wins).
    P.append((r"(?<![A-Za-z0-9_])K_rhs(?![A-Za-z0-9_])", "trK_rhs"))
    P.append((r"(?<![A-Za-z0-9_])K(?![A-Za-z0-9_])", "trK"))

    for c in range(3):
        P.append((rf"\bB_rhs{c}\b", f"gaugeB{c}_rhs"))
        P.append((rf"(?<![A-Za-z0-9_])B{c}(?![A-Za-z0-9_])", f"gaugeB{c}"))
        P.append((rf"\bGt_rhs{c}\b", f"CAP_Gt{c}_rhs"))
        P.append((rf"(?<![A-Za-z0-9_])Gt{c}(?![A-Za-z0-9_])", f"CAP_Gt{c}"))
        P.append((rf"\bb_rhs{c}\b", f"beta{c}_rhs"))
        P.append((rf"(?<![A-Za-z0-9_])b{c}(?![A-Za-z0-9_])", f"beta{c}"))

    # a_rhs -> alpha_rhs; lambda_f -> lf
    P.append((r"\ba_rhs\b", "alpha_rhs"))
    P.append((r"\blambda_f\b", "lf"))

    # eta -> etadamp  (BSSN scalar; EMDA reserves eta[] for matter sector,
    # so don't touch eta[..] indexing or the existing 'etadamp' identifier).
    P.append((r"(?<![A-Za-z0-9_])eta(?![A-Za-z0-9_\[])", "etadamp"))


_build_patterns()


def translate_bssn_to_emda_name(name: str) -> str:
    """Apply the EMDA rename pattern set to a single Symbol name string.
    Operates on string symbol names rather than emitted text, so [pp]-suffixed
    leaves (`K[pp]`, `gt0[pp]`, `grad_0_At3[pp]`) translate correctly."""
    import re
    out = name
    for pat, repl in _BSSN_TO_EMDA_PATTERNS:
        out = re.sub(pat, repl, out)
    return out


def bssn_to_emda_symbol_renames(symbols: Iterable[sym.Symbol]) -> Dict[str, str]:
    """Build a {old_name: new_name} dict for the given set of free symbols,
    skipping any that the pattern set leaves unchanged."""
    out: Dict[str, str] = {}
    for s in symbols:
        old = str(s)
        new = translate_bssn_to_emda_name(old)
        if new != old:
            out[old] = new
    return out


def bssn_to_emda_key_renames(keys: Iterable[str]) -> Dict[str, str]:
    """Build a {old_key: new_key} dict for chunk-output keys using the same
    rule set as the symbol rename. Distinguishes scalar/vector output names
    (a_rhs -> alpha_rhs, Gt_rhs0 -> CAP_Gt_rhs0, gt_rhs00 -> gt00_rhs, ...)."""
    out: Dict[str, str] = {}
    for k in keys:
        new = translate_bssn_to_emda_name(k)
        if new != k:
            out[k] = new
    return out


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def _selftest() -> None:
    print("=== cascade_dialect self-test ===")

    # BSSN dialect 2-digit
    s = BSSN_DIALECT.spec("gt")
    assert s.component(0, 0) == "gt00"
    assert s.component(1, 2) == s.component(2, 1) == "gt12"
    assert BSSN_DIALECT.spec("Gt").component(2) == "Gt2"
    assert BSSN_DIALECT.spec("K").component() == "K"
    print("  BSSN dialect: ok")

    # EMDA dialect
    assert EMDA_DIALECT.spec("Gt").component(0) == "CAP_Gt0"
    assert EMDA_DIALECT.spec("B").component(1) == "gaugeB1"
    assert EMDA_DIALECT.spec("K").component() == "trK"
    assert EMDA_DIALECT.spec("a").component() == "alpha"
    assert EMDA_DIALECT.spec("eta").component() == "etadamp"
    assert EMDA_DIALECT.spec("perpE").component(2) == "perpE2"
    print("  EMDA dialect: ok")

    # Rename via the regex-based pattern set (the real path emda_cascade uses).
    # Use [pp]-suffixed leaves to mirror what the BSSN module actually emits.
    syms = sym.symbols("K[pp] alpha[pp] B0[pp] Gt1[pp] eta gt5[pp] At3[pp]")
    renames = bssn_to_emda_symbol_renames(syms)
    expected = {
        "K[pp]":    "trK[pp]",
        "B0[pp]":   "gaugeB0[pp]",
        "Gt1[pp]":  "CAP_Gt1[pp]",
        "eta":      "etadamp",
        "gt5[pp]":  "gt22[pp]",
        "At3[pp]":  "At11[pp]",
    }
    assert renames == expected, f"got: {renames}\nwant: {expected}"
    print(f"  symbol-name renames: {renames}")

    # Key renames for output dict keys (RHS suffix-position swap, scalar names).
    key_renames = bssn_to_emda_key_renames([
        "a_rhs", "K_rhs", "Gt_rhs0", "B_rhs2", "b_rhs1",
        "gt_rhs00", "gt_rhs22", "At_rhs01",
        "gt00",      # unchanged (already 2-digit)
        "chi_rhs",   # unchanged
    ])
    assert key_renames == {
        "a_rhs": "alpha_rhs", "K_rhs": "trK_rhs",
        "Gt_rhs0": "CAP_Gt0_rhs", "B_rhs2": "gaugeB2_rhs",
        "b_rhs1": "beta1_rhs",
        "gt_rhs00": "gt00_rhs", "gt_rhs22": "gt22_rhs",
        "At_rhs01": "At01_rhs",
    }, key_renames
    print(f"  key renames ({len(key_renames)} keys): ok")

    # Flatten helpers under each dialect
    mat = sym.Matrix([[sym.Symbol(f"x{i}{j}") for j in range(3)] for i in range(3)])
    bssn_keys = list(flatten_sym33(mat, "gt", BSSN_DIALECT).keys())
    assert bssn_keys == ["gt00", "gt01", "gt02", "gt11", "gt12", "gt22"], bssn_keys
    emda_keys = list(flatten_sym33(mat, "At_rhs", EMDA_DIALECT).keys())
    assert emda_keys == ["At_rhs00", "At_rhs01", "At_rhs02",
                         "At_rhs11", "At_rhs12", "At_rhs22"], emda_keys
    print("  flatten helpers: ok")

    # Derivative-family specs (array-access naming for the looped emitter).
    gt_spec = make_sym33_2digit("gt")
    dgt = make_deriv_family("gt", gt_spec, order=1)
    assert dgt.rank == 3 and dgt.symmetric
    assert dgt.component(0, 1, 1) == "d_gt[0][1][1]"
    assert dgt.component(2, 1, 2) == "d_gt[2][1][2]"
    d2gt = make_deriv_family("gt", gt_spec, order=2)
    assert d2gt.rank == 4
    assert d2gt.component(0, 1, 2, 2) == "d2_gt[0][1][2][2]"
    dal = make_deriv_family("a", make_scalar("a"), order=1, array="d_al")
    assert dal.rank == 1 and dal.component(2) == "d_al[2]"
    print("  derivative-family specs: ok")

    print("=== all self-tests passed ===")


if __name__ == "__main__":
    _selftest()
