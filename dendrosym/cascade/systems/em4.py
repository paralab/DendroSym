"""em4.py -- Maxwell + GLM divergence-cleaning (EM4) RHS: SymPy spec.

Pointwise RHS operator for the EM4 system as defined in em4_base/em4.py
(dendrosym). Eight evolution variables (E0-2, B0-2, Phi, Psi) with hyperbolic
(GLM) divergence cleaning. The RHS is FIRST-ORDER: every term is a first
derivative of a field (curl/div/grad) plus a damping/source term -- there are
NO products, no second derivatives, and crucially NO shared intermediates
between outputs. It is therefore the extreme low-peak-live / direct-linear-map
end of the cascade catalog: a single chunk, nothing to layer, ~zero CSE
opportunity. We include it as the simple-system control point and to exercise
the SIMD emitter on a first-order system.

Equations (i in {0,1,2}; eps = Levi-Civita):
    B_rhs[i]  = -(curl E)_i + d_i Phi
    E_rhs[i]  =  (curl B)_i - 4*pi*J_i - d_i Psi
    Phi_rhs   =  div B - kappa_2 * Phi
    Psi_rhs   =  4*pi*rho_e - div E - kappa_1 * Psi

Leaves (per_point=True): first-derivative arrays grad_{d}_{field}[pp] for the
8 fields (24), the undifferentiated Phi[pp]/Psi[pp] (damping), the sources
J0/J1/J2[pp] and rho_e[pp], and the scalar params kappa_1/kappa_2.
"""
from collections import OrderedDict
import math
import sympy as sym

_PI = sym.Float(math.pi)


def em4_rhs_spec(per_point: bool = True):
    """EM4 (Maxwell + GLM cleaning) RHS as a single-chunk named cascade.

    Returns (chunks, leaves) in the form build_cascade_ir expects.
    """
    sfx = "[pp]" if per_point else ""

    def g(d, f):  # first derivative leaf grad_<d>_<field>
        return sym.Symbol(f"grad_{d}_{f}{sfx}", real=True)

    # derivative leaves for every field
    fields = ["E0", "E1", "E2", "B0", "B1", "B2", "Phi", "Psi"]
    grads = {(d, f): g(d, f) for d in (0, 1, 2) for f in fields}

    # undifferentiated fields needed (damping terms only)
    Phi = sym.Symbol(f"Phi{sfx}", real=True)
    Psi = sym.Symbol(f"Psi{sfx}", real=True)
    # sources
    J0 = sym.Symbol(f"J0{sfx}", real=True)
    J1 = sym.Symbol(f"J1{sfx}", real=True)
    J2 = sym.Symbol(f"J2{sfx}", real=True)
    rho_e = sym.Symbol(f"rho_e{sfx}", real=True)
    # scalar params (bare names -> VSET broadcast in the SIMD emitter)
    kappa_1 = sym.Symbol("kappa_1", real=True)
    kappa_2 = sym.Symbol("kappa_2", real=True)

    leaves = set(grads.values()) | {Phi, Psi, J0, J1, J2, rho_e,
                                    kappa_1, kappa_2}

    G = lambda d, f: grads[(d, f)]
    # curl components: (curl V)_i = eps_ijk d_j V_k
    curlE = [G(1, "E2") - G(2, "E1"),
             G(2, "E0") - G(0, "E2"),
             G(0, "E1") - G(1, "E0")]
    curlB = [G(1, "B2") - G(2, "B1"),
             G(2, "B0") - G(0, "B2"),
             G(0, "B1") - G(1, "B0")]
    divE = G(0, "E0") + G(1, "E1") + G(2, "E2")
    divB = G(0, "B0") + G(1, "B1") + G(2, "B2")
    J = [J0, J1, J2]
    gradPhi = [G(0, "Phi"), G(1, "Phi"), G(2, "Phi")]
    gradPsi = [G(0, "Psi"), G(1, "Psi"), G(2, "Psi")]
    four_pi = 4 * _PI

    suf = "_out" if per_point else ""
    rhs = OrderedDict()
    for i in range(3):
        rhs[f"B_rhs{i}{suf}"] = -curlE[i] + gradPhi[i]
    for i in range(3):
        rhs[f"E_rhs{i}{suf}"] = curlB[i] - four_pi * J[i] - gradPsi[i]
    rhs[f"Phi_rhs{suf}"] = divB - kappa_2 * Phi
    rhs[f"Psi_rhs{suf}"] = four_pi * rho_e - divE - kappa_1 * Psi

    # one chunk: EM4 has no shared intermediates to layer.
    chunks = [("rhs", rhs)]
    return chunks, leaves


if __name__ == "__main__":
    chunks, leaves = em4_rhs_spec()
    print(f"leaves ({len(leaves)}): {sorted(s.name for s in leaves)}\n")
    for name, outputs in chunks:
        print(f"chunk {name!r}:")
        for k, v in outputs.items():
            print(f"  {k} = {v}")
