"""bssn_clean.py -- a friendly, readable BSSN cascade definition (list style).

Same engine as bssn_cascade.py (cascade_builder + cascade_emit, unchanged), but
written as a clear example: compute physics, list the named chunks, build. The
physics lives in `compute_physics`; the chunk declaration is a short list. A
tiny `Cascade` helper hides flattening / leaf bookkeeping / naming.

Drop-in: `build_ir(...)` returns the same CascadeResult as
bssn_cascade.build_ir(...). The __main__ self-test asserts identical
(specs, leaves) for every gauge config, so this needs no re-runs to trust.

`bssn_decorated.py` is the SAME thing in a decorator style, reusing
`compute_physics` + `Cascade` from here.
"""

import sympy as sym
from collections import OrderedDict

from dendrosym.cascade.builder import (
    build_cascade_ir, flatten_sym33, flatten_tensor3, flatten_vec3, E_IJ_SYM,
)
from dendrosym.cascade.systems.bssn.physics import compute_physics, RHS_OUTPUT_NAMES


# --- field tags: make `R=sym33(R)` read better than `R=("sym33", R)` ---
def sym33(m): return ("sym33", m)
def t3(t):    return ("tensor3", t)
def vec(v):   return ("vec", v)

_FLATTEN = {"sym33": flatten_sym33, "tensor3": flatten_tensor3, "vec": flatten_vec3}


class Cascade:
    """Collect named chunks, then hand (specs, leaves) to the engine.

    chunk(name, **fields): each field is `prefix=value` where value is either a
        tagged tensor (sym33/t3/vec -> flattened to prefix00.. names) or a plain
        scalar expr (-> one output named `prefix`). A ready {name: expr} dict can
        be splatted in directly: `chunk("rhs_assembly", **rhs_dict)`.
    ref(name): cross-chunk output referenced BY NAME -- never by value (SymPy
        Mul-flattening can hide a by-value subtree and silently recompute it).
    """

    def __init__(self):
        self._chunks = []

    @staticmethod
    def ref(name):
        return sym.Symbol(name)

    def chunk(self, name, **fields):
        out = OrderedDict()
        for prefix, val in fields.items():
            if isinstance(val, tuple):          # tagged tensor -> flatten
                kind, tensor = val
                out.update(_FLATTEN[kind](tensor, prefix))
            else:                               # plain scalar expr
                out[prefix] = val
        self._chunks.append((name, out))

    def specs_and_leaves(self, rhs_exprs):
        leaves = set()
        for e in rhs_exprs:
            leaves |= e.free_symbols
        produced = {sym.Symbol(k) for _n, outs in self._chunks for k in outs}
        return self._chunks, leaves - produced




def c1_named(C1):
    """First Christoffel, symmetric in the last two indices (custom naming)."""
    return OrderedDict(((f"C1_{k}{i}{j}", C1[k, i, j])
                        for k in range(3) for i in range(3) for j in range(i, 3)))


def rhs_dict(p):
    od = OrderedDict()
    od["a_rhs"] = p.a_rhs
    for i in range(3):
        od[f"b_rhs{i}"] = p.b_rhs[i]
    for i, j in E_IJ_SYM:
        od[f"gt_rhs{i}{j}"] = p.gt_rhs[i, j]
    od["chi_rhs"] = p.chi_rhs
    for i, j in E_IJ_SYM:
        od[f"At_rhs{i}{j}"] = p.At_rhs[i, j]
    od["K_rhs"] = p.K_rhs
    for i in range(3):
        od[f"Gt_rhs{i}"] = p.Gt_rhs_list[i]
    for i in range(3):
        od[f"B_rhs{i}"] = p.B_rhs[i]
    return od


# ---------------------------------------------------------------------------
# The 7 BSSN chunks, declared plainly (list style).
# ---------------------------------------------------------------------------
def declare_bssn_chunks(c, p):
    c.chunk("inverse_metric",       igt=sym33(p.igt), chi_inv=1 / p.chi)
    c.chunk("first_christoffel",    **c1_named(p.C1))
    c.chunk("second_christoffel",   C2_=t3(p.C2))
    c.chunk("complete_christoffel", C3_=t3(p.C3))
    c.chunk("ricci",                R=sym33(p.R), CalGt=vec(p.CalGt))
    c.chunk("derived_quantities",
            At_UU=sym33(p.At_UU), AikAkj=sym33(p.AikAkj),
            DiDj_a=sym33(p.DiDj_a), tf=sym33(p.tf),
            At_sqr=p.At_sqr, lap_a=p.lap_a)
    c.chunk("rhs_assembly",         **rhs_dict(p))


def build_specs(gauge="standard", ssl=False, cahd=False, eta_mode="scalar"):
    p = compute_physics(gauge=gauge, ssl=ssl, cahd=cahd, eta_mode=eta_mode)
    c = Cascade()
    declare_bssn_chunks(c, p)
    specs, leaves = c.specs_and_leaves(p.all_rhs)
    assert frozenset(specs[-1][1].keys()) == RHS_OUTPUT_NAMES
    return specs, leaves


def build_ir(target_L=None, smart_split=None, gauge="standard", ssl=False,
             cahd=False, eta_mode="scalar", verbose=False):
    specs, leaves = build_specs(gauge=gauge, ssl=ssl, cahd=cahd, eta_mode=eta_mode)
    return build_cascade_ir(specs, leaves, target_L=target_L,
                            smart_split=smart_split, verbose=verbose)


def _selftest():
    from dendrosym.cascade.systems.bssn import cascade as bssn_cascade
    configs = [dict(gauge="standard"),
               dict(gauge="standard", ssl=True, cahd=True),
               dict(gauge="rochester"),
               dict(gauge="standard", cahd=True, eta_mode="array")]
    ok = True
    for cfg in configs:
        s_new, l_new = build_specs(**cfg)
        s_old, l_old = bssn_cascade.build_specs(**cfg)
        same = (l_new == l_old
                and [n for n, _ in s_new] == [n for n, _ in s_old]
                and all(list(on.items()) == list(oo.items())
                        for (_, on), (_, oo) in zip(s_new, s_old)))
        print(f"{'IDENTICAL' if same else 'MISMATCH':10s} {cfg}")
        ok = ok and same
    print("=== PASS ===" if ok else "=== FAIL ===")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
