"""GR-MHD source terms as a cascade spec.

Transcribed from D. Neilsen's `mhdsrc.F`, routine `mhdsrc_dens`: the pointwise
source terms of the GR-MHD evolution on a curved background. The primitives
(rho, v^i, P) arrive as inputs, so this is a pointwise algebraic map and fits
the construction; the conservative-to-primitive recovery, which is an iterative
root find, is outside it. The divergence-cleaning terms for dtB and dtPsi are
finite differences of neighbouring points rather than pointwise algebra and are
likewise excluded, leaving the four genuine source outputs.

Unlike EMDA, this is not the BSSN geometry sector with extra fields: it is a
different formulation (ADM variables, densitized conservatives) and a different
physical system, so it tests generality rather than extension.
"""
from collections import OrderedDict

import sympy as sym

E_IJ_SYM = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
IDX = "[pp]"


def _s(name):
    return sym.Symbol(name + IDX)


def build_grmhd_specs():
    # ---- inputs, named as in mhdsrc.F -------------------------------------
    gd = sym.Matrix(3, 3, lambda i, j: _s("g%d%d" % (min(i, j) + 1, max(i, j) + 1)))
    Kd = sym.Matrix(3, 3, lambda i, j: _s("K%d%d" % (min(i, j) + 1, max(i, j) + 1)))
    sdetg = _s("sdetg")
    Alpha = _s("Alpha")
    A = [_s("A1"), _s("A2"), _s("A3")]
    dBeta = sym.Matrix(3, 3, lambda i, j: _s(
        "d%sBeta%s" % ("xyz"[i], "xyz"[j])))          # dBeta[i,j] = d_i beta^j
    Scons = [_s("Sx"), _s("Sy"), _s("Sz")]
    Bcons = [_s("Bx"), _s("By"), _s("Bz")]
    vu_in = [_s("vx"), _s("vy"), _s("vz")]
    D, Tau, P = _s("D"), _s("Tau"), _s("P")
    # d_k g_ij, as the Fortran names them: d<k><i><j>
    dgd = {}
    for k in range(3):
        for i, j in E_IJ_SYM:
            dgd[(k, i, j)] = _s("d%d%d%d" % (k + 1, i + 1, j + 1))
            dgd[(k, j, i)] = dgd[(k, i, j)]

    specs = []

    # ---- L1: inverse 3-metric and the two reciprocals ---------------------
    detg = gd.det()
    adj = gd.adjugate()
    inv_out = OrderedDict()
    for i, j in E_IJ_SYM:
        inv_out["gu%d%d" % (i + 1, j + 1)] = adj[i, j] / detg
    inv_out["isdetg"] = 1 / sdetg
    specs.append(("inverse_metric", inv_out))
    gu = sym.Matrix(3, 3, lambda i, j: sym.Symbol(
        "gu%d%d" % (min(i, j) + 1, max(i, j) + 1)))
    isdetg = sym.Symbol("isdetg")

    # ---- L2: lowered Christoffel, linear in the metric derivatives --------
    Cd = {}
    cd_out = OrderedDict()
    for k in range(3):
        for i, j in E_IJ_SYM:
            e = sym.Rational(1, 2) * (dgd[(i, k, j)] + dgd[(j, k, i)] - dgd[(k, i, j)])
            cd_out["Cd%d%d%d" % (k + 1, i + 1, j + 1)] = e
    specs.append(("first_christoffel", cd_out))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                a, b = min(i, j), max(i, j)
                Cd[(k, i, j)] = sym.Symbol("Cd%d%d%d" % (k + 1, a + 1, b + 1))

    # ---- L3: raised Christoffel -------------------------------------------
    c_out = OrderedDict()
    C = {}
    for k in range(3):
        for i, j in E_IJ_SYM:
            c_out["C%d%d%d" % (k + 1, i + 1, j + 1)] = sum(
                gu[k, m] * Cd[(m, i, j)] for m in range(3))
    specs.append(("second_christoffel", c_out))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                a, b = min(i, j), max(i, j)
                C[(k, i, j)] = sym.Symbol("C%d%d%d" % (k + 1, a + 1, b + 1))

    # ---- L4: undensitize --------------------------------------------------
    und = OrderedDict()
    for i in range(3):
        und["Sd%d" % (i + 1)] = isdetg * Scons[i]
    for i in range(3):
        und["Bu%d" % (i + 1)] = isdetg * Bcons[i]
    specs.append(("undensitize", und))
    Sd = [sym.Symbol("Sd%d" % (i + 1)) for i in range(3)]
    Bu = [sym.Symbol("Bu%d" % (i + 1)) for i in range(3)]

    # ---- L5: raise and lower ----------------------------------------------
    rl = OrderedDict()
    for i in range(3):
        rl["Su%d" % (i + 1)] = sum(gu[i, m] * Sd[m] for m in range(3))
    for i in range(3):
        rl["vd%d" % (i + 1)] = sum(gd[i, m] * vu_in[m] for m in range(3))
    for i in range(3):
        rl["Bd%d" % (i + 1)] = sum(gd[i, m] * Bu[m] for m in range(3))
    specs.append(("raise_lower", rl))
    Su = [sym.Symbol("Su%d" % (i + 1)) for i in range(3)]
    vd = [sym.Symbol("vd%d" % (i + 1)) for i in range(3)]
    Bd = [sym.Symbol("Bd%d" % (i + 1)) for i in range(3)]

    # ---- L6: fluid scalars -------------------------------------------------
    sc = OrderedDict()
    sc["vsq"] = sum(vu_in[m] * vd[m] for m in range(3))
    specs.append(("vsq", sc))
    vsq = sym.Symbol("vsq")
    sc2 = OrderedDict()
    sc2["Wsq"] = 1 / (1 - vsq)
    sc2["Bv"] = sum(Bd[m] * vu_in[m] for m in range(3))
    specs.append(("fluid_scalars", sc2))
    Wsq, Bv = sym.Symbol("Wsq"), sym.Symbol("Bv")

    # ---- L7: stress tensor, mixed indices ----------------------------------
    pt = OrderedDict()
    for i in range(3):
        for j in range(3):
            e = Sd[j] * vu_in[i] - Bu[i] * Bd[j] / Wsq - Bv * Bu[i] * vd[j]
            if i == j:
                sgn = [-1, -1, -1]
                sgn[i] = 1
                e = (Sd[j] * vu_in[i] + P
                     - sym.Rational(1, 2) / Wsq * sum(sgn[m] * Bu[m] * Bd[m]
                                                      for m in range(3))
                     - sym.Rational(1, 2) * Bv * sum(sgn[m] * Bu[m] * vd[m]
                                                     for m in range(3)))
            pt["PTud%d%d" % (i + 1, j + 1)] = e
    specs.append(("stress_tensor", pt))
    PTud = sym.Matrix(3, 3, lambda i, j: sym.Symbol("PTud%d%d" % (i + 1, j + 1)))

    # ---- L8: extrinsic curvature, mixed ------------------------------------
    kd = OrderedDict()
    for i in range(3):
        for j in range(3):
            kd["Kdu%d%d" % (i + 1, j + 1)] = sum(gu[j, m] * Kd[i, m]
                                                 for m in range(3))
    specs.append(("Kdu", kd))
    Kdu = sym.Matrix(3, 3, lambda i, j: sym.Symbol("Kdu%d%d" % (i + 1, j + 1)))

    # ---- L9: lapse gradient -------------------------------------------------
    da = OrderedDict()
    for i in range(3):
        da["dAlpha%d" % (i + 1)] = Alpha * A[i]
    specs.append(("lapse_gradient", da))
    dAlpha = [sym.Symbol("dAlpha%d" % (i + 1)) for i in range(3)]

    # ---- L10: the sources ---------------------------------------------------
    out = OrderedDict()
    out["dtTau"] = (Alpha * sdetg * sum(PTud[i, j] * Kdu[i, j]
                                        for i in range(3) for j in range(3))
                    - sdetg * sum(Su[m] * dAlpha[m] for m in range(3)))
    for i in range(3):
        out["dtS%s" % "xyz"[i]] = (
            Alpha * sdetg * sum(C[(k, m, i)] * PTud[m, k]
                                for k in range(3) for m in range(3))
            + sdetg * sum(Sd[m] * dBeta[i, m] for m in range(3))
            - dAlpha[i] * (Tau + D))
    specs.append(("sources", out))

    leaves = set()
    for _n, o in specs:
        for e in o.values():
            leaves |= e.free_symbols
    leaves -= {sym.Symbol(k) for _n, o in specs for k in o.keys()}
    return specs, leaves
