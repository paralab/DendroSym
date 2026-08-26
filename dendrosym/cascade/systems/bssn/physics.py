"""bssn_physics.py -- the single source of truth for the BSSN RHS physics.

Extracted so `bssn_cascade.py` (original driver) and `bssn_clean.py` /
`bssn_decorated.py` (clean front-ends) share ONE copy of the equations instead
of duplicating them. The two drivers keep their OWN spec-assembly code, so the
bssn_clean self-test still cross-checks the two assemblies -- that's the lock.

Contents:
  compute_physics(...)      -> SimpleNamespace of all RHS quantities
  RHS_OUTPUT_NAMES          -> the 24 RHS output keys

Ricci now comes from the fixed dendro.compute_ricci (which re-includes the xRphi
conformal-factor term); the former _compute_ricci_correct workaround is deleted.
Numerics are unchanged -- byte-identical emitted C++ vs the pre-fix golden.
"""

from __future__ import annotations

import types
import sympy as sym
from sympy import Matrix, Rational
from collections import OrderedDict

from dendrosym.cascade.builder import E_IJ_SYM



def compute_physics(gauge="standard", ssl=False, cahd=False, eta_mode="scalar"):
    """Stock Dendro-GR BSSN RHS quantities, as a namespace consumed by the
    cascade drivers. See Concepts/BSSN Cascade vs Dendro-GR for what differs."""
    if gauge not in ("standard", "rochester"):
        raise ValueError(f"gauge must be 'standard'|'rochester', got {gauge!r}")
    if eta_mode not in ("scalar", "array"):
        raise ValueError(f"eta_mode must be 'scalar'|'array', got {eta_mode!r}")
    from dendrosym.cascade.systems.bssn.legacy import dendro
    from dendrosym.cascade.systems.bssn.legacy import bssn

    a, chi, K, Gt, b, B = bssn.a, bssn.chi, bssn.K, bssn.Gt, bssn.b, bssn.B
    gt, At, igt = bssn.gt, bssn.At, bssn.igt
    d, d2, ad, weight = bssn.d, bssn.d2, bssn.ad, bssn.weight
    l1, l2, l3, l4 = bssn.l1, bssn.l2, bssn.l3, bssn.l4
    lf0, lf1 = bssn.lf0, bssn.lf1
    eta = sym.Symbol("eta[pp]") if eta_mode == "array" else sym.Symbol("eta")

    C1 = dendro.get_first_christoffel()
    C2 = dendro.get_second_christoffel()
    C3 = dendro.get_complete_christoffel(chi)
    R, Rt, Rphi, CalGt = dendro.compute_ricci(Gt, chi)  # R now includes xRphi

    At_UU = dendro.up_up(At)
    AikAkj = Matrix([
        sum(At[i, k] * sum(dendro.inv_metric[k, l] * At[l, j] for l in dendro.e_i)
            for k in dendro.e_i)
        for i, j in dendro.e_ij
    ]).reshape(3, 3)
    DiDj_a = dendro.DiDj(a)
    tf = dendro.trace_free(a * R - DiDj_a)
    At_sqr = dendro.sqr(At)
    lap_a = dendro.laplacian(a, chi)

    a_rhs = l1 * dendro.lie(b, a) - 2 * a * K
    if ssl:
        W = sym.sqrt(chi)
        a_rhs += sym.Symbol("ssl_fac") * W * (a - W)

    if gauge == "rochester":
        xi2, xi3 = sym.Symbol("BSSN_XI[1]"), sym.Symbol("BSSN_XI[2]")
        b_rhs = [xi2 * dendro.vec_j_ad_j(b, b[i]) + Rational(3, 4) * xi3 * Gt[i]
                 - eta * b[i] for i in dendro.e_i]
    else:
        b_rhs = [Rational(3, 4) * (lf0 + lf1 * a) * B[i]
                 + l2 * dendro.vec_j_ad_j(b, b[i]) for i in dendro.e_i]

    gt_rhs = dendro.lie(b, gt, weight) - 2 * a * At
    chi_rhs = dendro.lie(b, chi, weight) + Rational(2, 3) * (chi * a * K)
    if cahd:
        # cross-chunk outputs referenced BY NAME (not by value -- Mul-flatten).
        ham = (chi * sum(sym.Symbol(f"igt{min(j, k)}{max(j, k)}")
                         * sym.Symbol(f"R{min(j, k)}{max(j, k)}")
                         for j, k in dendro.e_ij)
               - sym.Symbol("At_sqr") + Rational(2, 3) * K ** 2)
        chi_rhs += sym.Symbol("cahd_coef") * chi * ham

    At_rhs = dendro.lie(b, At, weight) + chi * tf + a * (K * At - 2 * AikAkj)
    K_rhs = dendro.lie(b, K) - lap_a + a * (K * K / 3 + At_sqr)

    Gt_rhs = (
        Matrix([sum(b[j] * ad(j, Gt[i]) for j in dendro.e_i) for i in dendro.e_i])
        - Matrix([sum(CalGt[j] * d(j, b[i]) for j in dendro.e_i) for i in dendro.e_i])
        + Rational(2, 3) * Matrix([CalGt[i] * sum(d(j, b[j]) for j in dendro.e_i) for i in dendro.e_i])
        + Matrix([sum(igt[j, k] * d2(j, k, b[i]) + igt[i, j] * d2(j, k, b[k]) / 3
                      for j, k in dendro.e_ij) for i in dendro.e_i])
        - Matrix([sum(2 * At_UU[i, j] * d(j, a) for j in dendro.e_i) for i in dendro.e_i])
        + Matrix([sum(2 * a * dendro.C2[i, j, k] * At_UU[j, k] for j, k in dendro.e_ij) for i in dendro.e_i])
        - Matrix([sum(a * (3 / chi * At_UU[i, j] * d(j, chi)
                           + Rational(4, 3) * dendro.inv_metric[i, j] * d(j, K))
                      for j in dendro.e_i) for i in dendro.e_i])
    )
    Gt_rhs_list = [x for row in Gt_rhs.tolist() for x in row]

    if gauge == "rochester":
        B_rhs = [sym.Integer(0)] * 3
    else:
        B_rhs = [Gt_rhs_list[i] - eta * B[i] + l3 * dendro.vec_j_ad_j(b, B[i])
                 - l4 * dendro.vec_j_ad_j(b, Gt[i]) for i in dendro.e_i]

    all_rhs = ([a_rhs] + b_rhs + [gt_rhs[i, j] for i, j in E_IJ_SYM] + [chi_rhs]
               + [At_rhs[i, j] for i, j in E_IJ_SYM] + [K_rhs]
               + Gt_rhs_list + B_rhs)

    return types.SimpleNamespace(
        chi=chi, igt=igt, C1=C1, C2=C2, C3=C3, R=R, CalGt=CalGt,
        At_UU=At_UU, AikAkj=AikAkj, DiDj_a=DiDj_a, tf=tf, At_sqr=At_sqr, lap_a=lap_a,
        a_rhs=a_rhs, b_rhs=b_rhs, gt_rhs=gt_rhs, chi_rhs=chi_rhs, At_rhs=At_rhs,
        K_rhs=K_rhs, Gt_rhs_list=Gt_rhs_list, B_rhs=B_rhs, all_rhs=all_rhs)


# Output names written to harness arrays (vs intermediate temps).
RHS_OUTPUT_NAMES = frozenset(
    ["a_rhs", "chi_rhs", "K_rhs"]
    + [f"b_rhs{i}" for i in range(3)]
    + [f"gt_rhs{i}{j}" for i, j in E_IJ_SYM]
    + [f"At_rhs{i}{j}" for i, j in E_IJ_SYM]
    + [f"Gt_rhs{i}" for i in range(3)]
    + [f"B_rhs{i}" for i in range(3)]
)
