"""bssn_looped.py -- BSSN system definition for the generic looped emitter.

This is the BSSN-specific half of the structured looped emitter: it supplies
the cascade layers as `TensorEqn`s (regular tensor layers) plus the irregular
remainder as verbatim lines from the hand oracle (cascade_codegen). The generic
emitter (cascade_tensor_eqn) consumes this item list and knows nothing about
BSSN. Another system (mhd_looped, ...) supplies its own item list against the
same generic driver.

Hybrid coverage (locked design): regular tensor layers are looped from their
index structure; irregular layers (matrix-inverse L1, complete-Christoffel L4
with its Kronecker deltas, RHS assembly L7 with advection/gauge/SSL/CAHD/stores)
come from the oracle. As more layers move to TensorEqns the oracle remainder
shrinks. Currently generic: L2, L3, L5, L6.
"""

from __future__ import annotations

import sympy as sym

from dendrosym.cascade.tensor_eqn import TensorEqn, emit_looped_body


# ----------------------------------------------------------------------------
# Leaf-array prologue
# ----------------------------------------------------------------------------

def bssn_leaf_prologue(fused: bool = False) -> list:
    """Load BSSN state + derivative tensors into VEC arrays once per point,
    via the proven oracle prologue (symmetric state arrays gt[i][j] from the
    flat sym-6 layout; derivative arrays d_gt[d][i][j], d2_gt[k][l][i][j], ...
    that the looped layers consume)."""
    from dendrosym.cascade.systems.bssn.legacy import cascade_codegen as cc
    lines = list(cc.emit_unpack_state())
    lines.append("")
    lines += cc.emit_unpack_derivs(fused_derivs=fused)
    return lines


# ----------------------------------------------------------------------------
# BSSN cascade layers as TensorEqns (the index structure to capture)
# ----------------------------------------------------------------------------

def first_christoffel() -> TensorEqn:
    """L2: C1[k][i][j] = 1/2 (d_gt[j][k][i] + d_gt[i][k][j] - d_gt[k][i][j]),
    where d_gt[d][a][b] = d_d gt_{ab}. Symmetric in the last two indices (i,j)."""
    k, i, j = sym.symbols("k i j", integer=True)
    dgt = sym.IndexedBase("d_gt")
    return TensorEqn(
        out="C1", free=("k", "i", "j"),
        body=sym.Rational(1, 2) * (dgt[j, k, i] + dgt[i, k, j] - dgt[k, i, j]),
        symmetry=(("i", "j"),),
    )


def second_christoffel() -> TensorEqn:
    """L3: C2[i][j][k] = igt^{il} C1[l][j][k]  (contraction over l)."""
    i, j, k, l = sym.symbols("i j k l", integer=True)
    igt = sym.IndexedBase("igt")
    C1 = sym.IndexedBase("C1")
    return TensorEqn(
        out="C2", free=("i", "j", "k"),
        body=sym.Sum(igt[i, l] * C1[l, j, k], (l, 0, 2)),
    )


def ricci_eqns() -> list:
    """L5: conformal Ricci R[i][j], an ordered list of TensorEqns (regular
    contractions + scalar reductions). Mirrors oracle emit_L5. Leaves in scope:
    igt, gt, chi_inv (L1); C1 (L2), C2 (L3); d_Gt, d_ch, d2_ch, d2_gt (prologue).
    Downstream consumes CalGt and R."""
    i, j, k, l, m = sym.symbols("i j k l m", integer=True)
    igt = sym.IndexedBase("igt")
    gt = sym.IndexedBase("gt")
    C1 = sym.IndexedBase("C1")
    C2 = sym.IndexedBase("C2")
    dGt = sym.IndexedBase("d_Gt")
    dch = sym.IndexedBase("d_ch")
    d2ch = sym.IndexedBase("d2_ch")
    d2gt = sym.IndexedBase("d2_gt")
    CalGt = sym.IndexedBase("CalGt")
    Rt = sym.IndexedBase("Rt")
    C2_dc = sym.IndexedBase("C2_dc")
    xRphi = sym.IndexedBase("xRphi")
    chi_inv = sym.Symbol("chi_inv")
    half = sym.Rational(1, 2)

    eqns = []

    # CalGt^i = igt^{kl} C2[i][k][l]
    eqns.append(TensorEqn(
        out="CalGt", free=("i",),
        body=sym.Sum(igt[k, l] * C2[i, k, l], (k, 0, 2), (l, 0, 2))))

    # Rt[i][j] = -1/2 igt^{lm} d2gt_{lm,ij} + 1/2 (gt_{ki} d_j Gt^k + gt_{kj} d_i Gt^k)
    #            + 1/2 CalGt^k (C1_{ijk}+C1_{jik})
    #            + igt^{lm}(C2_{kli}C1_{jkm}+C2_{klj}C1_{ikm}+C2_{kim}C1_{klj})
    t1 = -half * sym.Sum(igt[l, m] * d2gt[l, m, i, j], (l, 0, 2), (m, 0, 2))
    t2 = half * sym.Sum(gt[k, i] * dGt[k, j] + gt[k, j] * dGt[k, i], (k, 0, 2))
    t3 = half * sym.Sum(CalGt[k] * (C1[i, j, k] + C1[j, i, k]), (k, 0, 2))
    t4 = sym.Sum(igt[l, m] * (C2[k, l, i] * C1[j, k, m]
                             + C2[k, l, j] * C1[i, k, m]
                             + C2[k, i, m] * C1[k, l, j]),
                 (k, 0, 2), (l, 0, 2), (m, 0, 2))
    eqns.append(TensorEqn(out="Rt", free=("i", "j"), body=t1 + t2 + t3 + t4,
                          symmetry=(("i", "j"),)))

    # C2_dc[i][j] = C2[k][j][i] d_k chi   (not symmetric)
    eqns.append(TensorEqn(
        out="C2_dc", free=("i", "j"),
        body=sym.Sum(C2[k, j, i] * dch[k], (k, 0, 2))))

    # Scalar chi factors.
    eqns.append(TensorEqn(out="chi_inv2", free=(), body=chi_inv ** 2))
    eqns.append(TensorEqn(out="half_chi_inv", free=(), body=half * chi_inv))
    eqns.append(TensorEqn(out="quarter_chi_inv2", free=(),
                          body=sym.Rational(1, 4) * sym.Symbol("chi_inv2")))

    # xRphi[i][j] = half_chi_inv (d2_ij chi - C2_dc_ij) - quarter_chi_inv2 d_i chi d_j chi
    eqns.append(TensorEqn(
        out="xRphi", free=("i", "j"),
        body=sym.Symbol("half_chi_inv") * (d2ch[i, j] - C2_dc[i, j])
             - sym.Symbol("quarter_chi_inv2") * dch[i] * dch[j],
        symmetry=(("i", "j"),)))

    # Scalar reductions for the conformal-factor Ricci trace part.
    eqns.append(TensorEqn(out="CalGt_dchi", free=(),
                          body=sym.Sum(CalGt[k] * dch[k], (k, 0, 2))))
    eqns.append(TensorEqn(out="igt_d2chi", free=(),
                          body=sym.Sum(igt[k, l] * d2ch[k, l], (k, 0, 2), (l, 0, 2))))
    eqns.append(TensorEqn(out="igt_dchi_dchi", free=(),
                          body=sym.Sum(igt[k, l] * dch[k] * dch[l], (k, 0, 2), (l, 0, 2))))
    eqns.append(TensorEqn(
        out="scalar_part", free=(),
        body=sym.Symbol("igt_d2chi")
             - sym.Rational(3, 2) * chi_inv * sym.Symbol("igt_dchi_dchi")
             - sym.Symbol("CalGt_dchi")))
    eqns.append(TensorEqn(out="half_chi_inv_scalar", free=(),
                          body=sym.Symbol("half_chi_inv") * sym.Symbol("scalar_part")))

    # R[i][j] = Rt[i][j] + xRphi[i][j] + half_chi_inv_scalar gt[i][j]
    eqns.append(TensorEqn(
        out="R", free=("i", "j"),
        body=Rt[i, j] + xRphi[i, j] + sym.Symbol("half_chi_inv_scalar") * gt[i, j],
        symmetry=(("i", "j"),)))

    return eqns


def derived_quantities_eqns() -> list:
    """L6: derived quantities (regular contractions + traces). Mirrors oracle
    emit_L6. Leaves in scope: igt, gt, At, al, ch (L1/state); C3 (L4); R (L5);
    d_al, d2_al, d_be (prologue). Consumed by L7: At_UU, AikAkj, DiDj_a, tf,
    At_sqr, lap_a, div_be."""
    i, j, k, l = sym.symbols("i j k l", integer=True)
    igt = sym.IndexedBase("igt")
    gt = sym.IndexedBase("gt")
    At = sym.IndexedBase("At")
    C3 = sym.IndexedBase("C3")
    R = sym.IndexedBase("R")
    Aup = sym.IndexedBase("Aup")
    At_UU = sym.IndexedBase("At_UU")
    DiDj_a = sym.IndexedBase("DiDj_a")
    d_al = sym.IndexedBase("d_al")
    d2_al = sym.IndexedBase("d2_al")
    d_be = sym.IndexedBase("d_be")
    al = sym.Symbol("al")
    ch = sym.Symbol("ch")

    eqns = []
    # At_UU = igt . At . igt  (symmetric)
    eqns.append(TensorEqn(
        out="At_UU", free=("i", "j"),
        body=sym.Sum(igt[i, k] * igt[j, l] * At[k, l], (k, 0, 2), (l, 0, 2)),
        symmetry=(("i", "j"),)))
    # Aup[k][j] = igt^{kl} At_{lj}  (full)
    eqns.append(TensorEqn(
        out="Aup", free=("k", "j"),
        body=sym.Sum(igt[k, l] * At[l, j], (l, 0, 2))))
    # AikAkj = At . Aup  (symmetric)
    eqns.append(TensorEqn(
        out="AikAkj", free=("i", "j"),
        body=sym.Sum(At[i, k] * Aup[k, j], (k, 0, 2)),
        symmetry=(("i", "j"),)))
    # DiDj_a = d2_al_{ij} - C3^l_{ij} d_l a   (symmetric)
    eqns.append(TensorEqn(
        out="DiDj_a", free=("i", "j"),
        body=d2_al[i, j] - sym.Sum(C3[l, i, j] * d_al[l], (l, 0, 2)),
        symmetry=(("i", "j"),)))
    # trace_val = igt^{ij}(a R_{ij} - DiDj_a_{ij})   (scalar)
    eqns.append(TensorEqn(
        out="trace_val", free=(),
        body=sym.Sum(igt[i, j] * (al * R[i, j] - DiDj_a[i, j]), (i, 0, 2), (j, 0, 2))))
    eqns.append(TensorEqn(out="tf_scalar", free=(),
                          body=sym.Rational(1, 3) * sym.Symbol("trace_val")))
    # tf_{ij} = a R_{ij} - DiDj_a_{ij} - gt_{ij} tf_scalar   (symmetric)
    eqns.append(TensorEqn(
        out="tf", free=("i", "j"),
        body=al * R[i, j] - DiDj_a[i, j] - gt[i, j] * sym.Symbol("tf_scalar"),
        symmetry=(("i", "j"),)))
    # At_sqr = At_{ij} At_UU^{ij}   (scalar)
    eqns.append(TensorEqn(
        out="At_sqr", free=(),
        body=sym.Sum(At[i, j] * At_UU[i, j], (i, 0, 2), (j, 0, 2))))
    # lap_a = chi igt^{ij} DiDj_a_{ij}   (scalar)
    eqns.append(TensorEqn(
        out="lap_a", free=(),
        body=sym.Sum(ch * igt[i, j] * DiDj_a[i, j], (i, 0, 2), (j, 0, 2))))
    # div_be = d_i beta^i   (scalar trace)
    eqns.append(TensorEqn(out="div_be", free=(),
                          body=sym.Sum(d_be[i, i], (i, 0, 2))))
    return eqns


# ----------------------------------------------------------------------------
# Item list + build entry point
# ----------------------------------------------------------------------------

def looped_items(simd: str = "avx2", ssl: bool = False, cahd: bool = False) -> list:
    """Ordered BSSN cascade items for the generic driver: regular layers as
    TensorEqns, irregular layers (L1/L4/L7) as oracle verbatim lines.

    ssl/cahd add the shock-avoiding-lapse and Hamiltonian-constraint-damping
    terms to L7 (gauge-independent regular layers are unchanged). The wrapper
    must define BSSN_ENABLE_SSL_HD and provide h_ssl/sig_ssl/t/dx_i/dt."""
    from dendrosym.cascade.systems.bssn.legacy import cascade_codegen as cc
    config = {"simd": simd, "ssl": ssl, "cahd": cahd}
    items: list = []
    items += bssn_leaf_prologue(fused=False)
    items.append("")
    items += cc.emit_L1()                                  # irregular: matrix inverse
    items.append("")
    items.append("// === L2: First Christoffel (GENERIC) ===")
    items.append(first_christoffel())
    items.append("")
    items.append("// === L3: Second Christoffel (GENERIC) ===")
    items.append(second_christoffel())
    items.append("")
    items += cc.emit_L4()                                  # irregular: Kronecker deltas
    items.append("")
    items.append("// === L5: Ricci tensor (GENERIC) ===")
    items += ricci_eqns()
    items.append("")
    items.append("// === L6: Derived quantities (GENERIC) ===")
    items += derived_quantities_eqns()
    items.append("")
    items += cc.emit_L7(config)                            # irregular: RHS assembly
    return items


def build(simd: str = "avx2", ssl: bool = False, cahd: bool = False,
          split: int = None) -> str:
    """Complete BSSN looped kernel body for include into the harness wrapper.

    split: FMA-accumulator fan-out for the generic layers. Default None picks
    2 for scalar (the serial FMA chain is latency-bound at width 1) and 1 for
    the SIMD widths (deployment default, unchanged)."""
    if split is None:
        split = 2 if simd == "scalar" else 1
    gauge = " + SSL/CAHD" if (ssl or cahd) else ""
    banner = [
        f"BSSN RHS via polynomial cascade, {simd.upper()}-batched (looped){gauge}.",
        "Regular layers (L2,L3,L5,L6) emitted generically from TensorEqns",
        "(cascade_tensor_eqn); irregular layers (L1,L4,L7) from the oracle.",
        "Included inside a harness wrapper that provides pointers + outer loops.",
    ]
    return emit_looped_body(looped_items(simd, ssl=ssl, cahd=cahd),
                            simd=simd, banner=banner, split=split)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--simd", choices=["scalar", "avx2", "avx512"], default="avx2")
    p.add_argument("--ssl", action="store_true", help="shock-avoiding lapse term")
    p.add_argument("--cahd", action="store_true", help="Ham-constraint damping")
    p.add_argument("--output", "-o", required=True)
    a = p.parse_args()
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    cpp = build(simd=a.simd, ssl=a.ssl, cahd=a.cahd)
    with open(a.output, "w") as f:
        f.write(cpp)
    print(f"wrote {len(cpp.splitlines())} lines to {a.output}")
