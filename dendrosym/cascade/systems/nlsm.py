"""nlsm.py -- NLSM (linear scalar wave) RHS spec for the cascade codegen, in the
em4.py style. RHS: chi_rhs = phi; phi_rhs = cx*grad2_0_0_chi + cy*grad2_1_1_chi +
cz*grad2_2_2_chi. A degenerate (1-layer, linear) system — shows the codegen handles
the trivial case; the real value of the cascade is complex tensor physics (BSSN)."""
from collections import OrderedDict
import sympy as sym


def nlsm_rhs_spec(per_point: bool = True):
    """Return (chunks, leaves) in the form build_cascade_ir expects."""
    sfx = "[pp]" if per_point else ""
    phi = sym.Symbol(f"phi{sfx}", real=True)
    g2 = {a: sym.Symbol(f"grad2_{a}_{a}_chi{sfx}", real=True) for a in (0, 1, 2)}
    cx, cy, cz = (sym.Symbol(s, real=True) for s in ("cx", "cy", "cz"))
    leaves = {phi} | set(g2.values()) | {cx, cy, cz}
    chunks = [("nlsm_rhs", OrderedDict([
        ("chi_rhs", phi),
        ("phi_rhs", cx * g2[0] + cy * g2[1] + cz * g2[2]),
    ]))]
    return chunks, leaves
