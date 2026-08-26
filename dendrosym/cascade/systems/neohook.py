"""neohook.py -- 2D Neo-Hookean hyperelastic stress: SymPy spec.

This is the talk's canonical worked example (slides 4-7). Inputs are the four
displacement-gradient components (a, b, c, d); outputs are the four entries
of the first Piola-Kirchhoff stress P_ij. Natural cascade depth is 4-5 layers.

    a = ∂u_0/∂X_0, b = ∂u_0/∂X_1, c = ∂u_1/∂X_0, d = ∂u_1/∂X_1

    F = I + ∇u                                  (layer 1, deg 1)
    C = F^T F                                   (layer 2, deg 2 in inputs)
    J = det F                                   (layer 2, deg 2)
    C^-1 = adj(C) / det(C)                      (layer 3)
    S = μ (I - C^-1) + λ ln(J) C^-1             (layer 4)
    P = F · S                                   (layer 5)

Material params μ (mu) and λ (lam) are scalar leaves.

Nothing here is BSSN-specific — we use the same primitives a 3D code would
(SymPy Matrix algebra), and the resulting CascadeResult flows through the
same emit pipeline as the BSSN cascade. That's the Phase 3 point.
"""

import sympy as sym
from sympy import Symbol, Matrix, Rational, log


def neohook_2d_spec(per_point=True):
    """Build the 2D Neo-Hookean cascade as named SymPy tensors.

    Parameters
    ----------
    per_point : bool, default True
        If True, the four ∇u inputs (a..d) and the four P outputs are named
        `a[pp]..d[pp]` and `P00_out..P11_out`. The cascade_emit kernel
        generator picks these up automatically as VLOAD'd arrays / VSTORE'd
        outputs. If False, bare names are used (matches the scalar bench
        that declares these as locals per iteration).

    Returns
    -------
    chunks : list of (chunk_name, OrderedDict[output_name, sym.Expr])
    leaves : set of free Symbols
        Order matches build_cascade_ir(chunks, leaves, ...).
    """
    from collections import OrderedDict

    if per_point:
        # `[pp]` suffix — the auto-classifier in cascade_emit puts these in
        # the pp_arrays bucket and emits VLOAD(arr+pp) inside the kernel.
        a = sym.Symbol('a[pp]', real=True)
        b = sym.Symbol('b[pp]', real=True)
        c = sym.Symbol('c[pp]', real=True)
        d = sym.Symbol('d[pp]', real=True)
        out_names = ("P00_out", "P01_out", "P10_out", "P11_out")
    else:
        a, b, c, d = sym.symbols('a b c d', real=True)
        out_names = ("P00", "P01", "P10", "P11")
    mu, lam = sym.symbols('mu lam', real=True, positive=True)

    leaves = {a, b, c, d, mu, lam}

    # ---- Layer 1: F = I + ∇u (4 entries, all linear in inputs) ----
    F = Matrix([
        [1 + a, b],
        [c, 1 + d],
    ])
    L1_outputs = OrderedDict([
        ("F00", F[0, 0]),
        ("F01", F[0, 1]),
        ("F10", F[1, 0]),
        ("F11", F[1, 1]),
    ])

    # Symbol references for layer 2's expressions to use.
    # The CascadeBuilder maps the chunk's output names to fresh Symbols of the
    # same name, so referencing `Symbol('F00')` in a later chunk picks up the
    # named output from layer 1 cleanly.
    sF = Matrix([
        [Symbol('F00'), Symbol('F01')],
        [Symbol('F10'), Symbol('F11')],
    ])

    # ---- Layer 2: C = F^T F (symmetric 3 entries) and J = det F ----
    C = sF.T * sF
    # We pull the 3 unique entries (00, 01, 11) since C is symmetric.
    C00 = C[0, 0]
    C01 = C[0, 1]
    C11 = C[1, 1]
    # J = det F. SymPy's det is just F00*F11 - F01*F10.
    J = sF.det()

    L2_outputs = OrderedDict([
        ("C00", C00),
        ("C01", C01),
        ("C11", C11),
        ("J", J),
    ])

    sC00 = Symbol('C00')
    sC01 = Symbol('C01')
    sC11 = Symbol('C11')
    sJ = Symbol('J')

    # ---- Layer 3: C^-1 (symmetric 3 entries) and log(J) ----
    # 2x2 inverse: detC = C00*C11 - C01^2; Cinv = (1/detC) * adj(C)
    detC = sC00 * sC11 - sC01 * sC01
    detC_inv = 1 / detC
    Cinv00 = detC_inv * sC11
    Cinv01 = -detC_inv * sC01
    Cinv11 = detC_inv * sC00
    log_J = log(sJ)

    L3_outputs = OrderedDict([
        ("Cinv00", Cinv00),
        ("Cinv01", Cinv01),
        ("Cinv11", Cinv11),
        ("logJ", log_J),
    ])

    sCinv00 = Symbol('Cinv00')
    sCinv01 = Symbol('Cinv01')
    sCinv11 = Symbol('Cinv11')
    slogJ = Symbol('logJ')

    # ---- Layer 4: S = μ (I - C^-1) + λ ln(J) C^-1 ----
    # Symmetric 3 entries.
    S00 = mu * (1 - sCinv00) + lam * slogJ * sCinv00
    S01 = mu * (0 - sCinv01) + lam * slogJ * sCinv01
    S11 = mu * (1 - sCinv11) + lam * slogJ * sCinv11

    L4_outputs = OrderedDict([
        ("S00", S00),
        ("S01", S01),
        ("S11", S11),
    ])

    sS = Matrix([
        [Symbol('S00'), Symbol('S01')],
        [Symbol('S01'), Symbol('S11')],
    ])

    # ---- Layer 5: P = F · S (4 entries; not symmetric) ----
    P = sF * sS
    L5_outputs = OrderedDict([
        (out_names[0], P[0, 0]),
        (out_names[1], P[0, 1]),
        (out_names[2], P[1, 0]),
        (out_names[3], P[1, 1]),
    ])

    chunks = [
        ("F", L1_outputs),
        ("C_J", L2_outputs),
        ("Cinv_logJ", L3_outputs),
        ("S", L4_outputs),
        ("P", L5_outputs),
    ]

    return chunks, leaves


def neohook_3d_spec(per_point=True):
    """Build the 3D Neo-Hookean cascade as named SymPy tensors.

    Inputs are the 9 components of ∇u (du_i/dX_j for i,j in 0..2). Outputs
    are the 9 entries of P_ij = F_ik S_kj. C, C^-1, S are symmetric (6 unique
    entries each).

    Same structure as 2D but with 3x3 tensors. Uses the cofactor / det
    formula for C^-1 (no general matrix inverse) so the kernel stays
    manageable.
    """
    from collections import OrderedDict
    if per_point:
        # 9 ∇u components named gradu_ij[pp]
        leaf_names = [f"gradu_{i}{j}" for i in range(3) for j in range(3)]
        gradu_syms = [sym.Symbol(f"{n}[pp]", real=True) for n in leaf_names]
        out_names = [f"P{i}{j}_out" for i in range(3) for j in range(3)]
    else:
        gradu_syms = sym.symbols(' '.join(f'gradu_{i}{j}' for i in range(3) for j in range(3)), real=True)
        out_names = [f"P{i}{j}" for i in range(3) for j in range(3)]
    g00, g01, g02, g10, g11, g12, g20, g21, g22 = gradu_syms
    mu, lam = sym.symbols('mu lam', real=True, positive=True)
    leaves = set(gradu_syms) | {mu, lam}

    # ---- Layer 1: F = I + ∇u (9 entries) ----
    F = Matrix([
        [1 + g00, g01, g02],
        [g10, 1 + g11, g12],
        [g20, g21, 1 + g22],
    ])
    L1_outputs = OrderedDict([
        (f"F{i}{j}", F[i, j]) for i in range(3) for j in range(3)
    ])

    sF = Matrix([[sym.Symbol(f"F{i}{j}") for j in range(3)] for i in range(3)])

    # ---- Layer 2: C = F^T F (6 sym entries) and J = det F ----
    C = sF.T * sF
    L2_outputs = OrderedDict()
    for i in range(3):
        for j in range(i, 3):
            L2_outputs[f"C{i}{j}"] = C[i, j]
    L2_outputs["J"] = sF.det()

    sC = Matrix([
        [sym.Symbol("C00"), sym.Symbol("C01"), sym.Symbol("C02")],
        [sym.Symbol("C01"), sym.Symbol("C11"), sym.Symbol("C12")],
        [sym.Symbol("C02"), sym.Symbol("C12"), sym.Symbol("C22")],
    ])
    sJ = sym.Symbol("J")

    # ---- Layer 3: C^-1 (6 sym entries) and log(J) ----
    # 3x3 sym inverse via cofactor / det
    detC = (
        sC[0,0]*(sC[1,1]*sC[2,2] - sC[1,2]*sC[1,2])
        - sC[0,1]*(sC[0,1]*sC[2,2] - sC[1,2]*sC[0,2])
        + sC[0,2]*(sC[0,1]*sC[1,2] - sC[1,1]*sC[0,2])
    )
    detC_inv = 1 / detC
    Cinv = OrderedDict()
    Cinv["Cinv00"] = detC_inv * (sC[1,1]*sC[2,2] - sC[1,2]*sC[1,2])
    Cinv["Cinv01"] = detC_inv * (sC[0,2]*sC[1,2] - sC[0,1]*sC[2,2])
    Cinv["Cinv02"] = detC_inv * (sC[0,1]*sC[1,2] - sC[0,2]*sC[1,1])
    Cinv["Cinv11"] = detC_inv * (sC[0,0]*sC[2,2] - sC[0,2]*sC[0,2])
    Cinv["Cinv12"] = detC_inv * (sC[0,2]*sC[0,1] - sC[0,0]*sC[1,2])
    Cinv["Cinv22"] = detC_inv * (sC[0,0]*sC[1,1] - sC[0,1]*sC[0,1])
    L3_outputs = OrderedDict(Cinv)
    L3_outputs["logJ"] = sym.log(sJ)

    sCinv = Matrix([
        [sym.Symbol("Cinv00"), sym.Symbol("Cinv01"), sym.Symbol("Cinv02")],
        [sym.Symbol("Cinv01"), sym.Symbol("Cinv11"), sym.Symbol("Cinv12")],
        [sym.Symbol("Cinv02"), sym.Symbol("Cinv12"), sym.Symbol("Cinv22")],
    ])
    slogJ = sym.Symbol("logJ")

    # ---- Layer 4: S = μ(I - C^-1) + λ ln(J) C^-1 (6 sym entries) ----
    L4_outputs = OrderedDict()
    for i in range(3):
        for j in range(i, 3):
            delta = 1 if i == j else 0
            L4_outputs[f"S{i}{j}"] = mu*(delta - sCinv[i,j]) + lam*slogJ*sCinv[i,j]

    sS = Matrix([
        [sym.Symbol("S00"), sym.Symbol("S01"), sym.Symbol("S02")],
        [sym.Symbol("S01"), sym.Symbol("S11"), sym.Symbol("S12")],
        [sym.Symbol("S02"), sym.Symbol("S12"), sym.Symbol("S22")],
    ])

    # ---- Layer 5: P = F · S (9 entries; not symmetric) ----
    P = sF * sS
    L5_outputs = OrderedDict([
        (out_names[3*i + j], P[i, j]) for i in range(3) for j in range(3)
    ])

    chunks = [
        ("F", L1_outputs),
        ("C_J", L2_outputs),
        ("Cinv_logJ", L3_outputs),
        ("S", L4_outputs),
        ("P", L5_outputs),
    ]

    return chunks, leaves


if __name__ == "__main__":
    import sys
    spec_fn = neohook_3d_spec if "--3d" in sys.argv else neohook_2d_spec
    chunks, leaves = spec_fn()
    print(f"leaves: {sorted(s.name for s in leaves)}")
    print()
    for name, outputs in chunks:
        print(f"chunk {name!r}:")
        for k, v in outputs.items():
            print(f"  {k} = {v}")
        print()
