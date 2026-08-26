"""emda_matter_source.py -- emit matter-source contributions to BSSN RHS.

The full EMDA-BSSN evolution equations (in emdabssn_eqns_configs.py) are
"vacuum BSSN + matter source terms in three places":

    trK_rhs       += 4*pi*alpha*(rho + perpTTrace)                  (line ~760)
    At_{ij}_rhs   += chi * (-8*pi*alpha) * trace_free(perpT)_{ij}   (line ~714)
    CAP_Gt_i_rhs  += -16*pi*alpha/chi * stressCurrent[i]            (line ~850)

vikr's BSSN cascade computes only the vacuum-BSSN part. To make a
machine-precision cascade for EMDA, we need to overlay these matter-source
contributions onto the cascade output. This script generates a single
`.cpp.inc` of `+=` statements that does that.

Approach:
  1. Import `emdabssn_eqns_configs` (it already builds `perpT`, `rho`,
     `stressCurrent`, `perpTTrace` symbolically at module scope).
  2. Build `trace_free(perpT) = perpT - (1/3) gt * (igt^kl perpT_kl)`.
  3. Form the three RHS contributions above.
  4. Substitute `Derivative(field(xx_temp,...), xx_temp)` -> `grad_0_field[pp]`
     and bare `field(xx_temp,...)` -> `field[pp]` so the output is harness-
     scope-friendly.
  5. SymPy CSE the 10 expressions (trK + 6 At + 3 CAP_Gt) together to share
     temps; emit declarations + `+=` assignments.

Output: harness_emda/src/gencode/emda_matter_source.cpp.inc

Usage:
    python emda_matter_source.py --output ../harness_emda/src/gencode/emda_matter_source.cpp.inc
"""

from __future__ import annotations

import argparse
import os
import sys

# Make emdabssn_eqns_configs importable.
EMDA_GR = os.environ.get("DENDRO_EMDA_GR", os.path.expanduser("~/research/emda-gr"))
if EMDA_GR not in sys.path:
    sys.path.insert(0, EMDA_GR)

import sympy as sym  # noqa: E402
from sympy import Rational, pi  # noqa: E402

import emdabssn_eqns_configs as E  # noqa: E402


_COORD_TO_DIR = {"xx_temp": 0, "yy_temp": 1, "zz_temp": 2}


def replace_derivs_and_funcs(expr: sym.Expr) -> sym.Expr:
    """Replace dendrosym's `Derivative(field(xx_temp, ...), xx_temp[, ...])`
    atoms with named `grad_d_field[pp]` (or `grad2_i_j_field[pp]`) symbols,
    and bare `field(xx_temp, ...)` calls with `field[pp]` symbols, so the
    expression can be ccode'd to harness-scope C++.

    Math functions like `exp`, `sqrt` are left intact (they're recognized
    by the C printer).
    """
    # Derivatives first (they must be replaced before their inner functions).
    derivs = sorted(expr.atoms(sym.Derivative), key=lambda d: -len(d.args))
    for d in derivs:
        fld = d.expr.func.__name__
        dirs = [_COORD_TO_DIR[str(v)] for v in d.variables]
        if len(dirs) == 1:
            new = sym.Symbol(f"grad_{dirs[0]}_{fld}[pp]")
        elif len(dirs) == 2:
            i, j = sorted(dirs)
            new = sym.Symbol(f"grad2_{i}_{j}_{fld}[pp]")
        else:
            raise RuntimeError(f"Unexpected deriv order: {d}")
        expr = expr.xreplace({d: new})

    # UndefinedFunction calls (state-variable evaluations).
    funcs = list(expr.atoms(sym.Function))
    safe_math = {"exp", "sqrt", "log", "sin", "cos", "tan", "Abs"}
    for f in funcs:
        if f.func.__name__ in safe_math:
            continue
        if isinstance(f, sym.Derivative):
            continue
        # Make sure args are the coord triple (xx_temp, yy_temp, zz_temp).
        # Anything else means we mis-recognized.
        if not all(str(a) in _COORD_TO_DIR for a in f.args):
            continue
        fld = f.func.__name__
        new = sym.Symbol(f"{fld}[pp]")
        expr = expr.xreplace({f: new})
    return expr


def build_matter_sources():
    """Return list of (lhs, expr) for the 10 += assignments."""
    gt = E.gt
    igt = E.igt
    perpT = E.perpT  # 3x3 sym.Matrix
    rho = E.rho
    perpTTrace = E.perpTTrace
    stressCurrent = E.stressCurrent
    alpha = E.alpha  # symbolic UndefinedFunction
    chi = E.chi

    # trace_free(perpT) = perpT - (1/3) gt * (igt^kl perpT_kl)
    trace_perpT = sum(igt[i, j] * perpT[i, j] for i in range(3)
                      for j in range(3))
    tf_perpT = sym.Matrix([
        [perpT[i, j] - Rational(1, 3) * gt[i, j] * trace_perpT
         for j in range(3)]
        for i in range(3)
    ])

    trK_src = 4 * pi * alpha * (rho + perpTTrace)
    At_src = sym.Matrix([
        [chi * (-8 * pi * alpha) * tf_perpT[i, j] for j in range(3)]
        for i in range(3)
    ])
    Gt_src = [-16 * pi * alpha / chi * stressCurrent[i] for i in range(3)]

    # Symmetric At indices: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2).
    # gaugeB_rhs[i] = CAP_Gt_rhs[i] + (terms not involving matter), so the
    # same matter delta propagates 1:1 from CAP_Gt onto gaugeB.
    return [
        ("trK_rhs[pp]",      trK_src),
        ("At00_rhs[pp]",     At_src[0, 0]),
        ("At01_rhs[pp]",     At_src[0, 1]),
        ("At02_rhs[pp]",     At_src[0, 2]),
        ("At11_rhs[pp]",     At_src[1, 1]),
        ("At12_rhs[pp]",     At_src[1, 2]),
        ("At22_rhs[pp]",     At_src[2, 2]),
        ("CAP_Gt0_rhs[pp]",  Gt_src[0]),
        ("CAP_Gt1_rhs[pp]",  Gt_src[1]),
        ("CAP_Gt2_rhs[pp]",  Gt_src[2]),
        ("gaugeB0_rhs[pp]",  Gt_src[0]),
        ("gaugeB1_rhs[pp]",  Gt_src[1]),
        ("gaugeB2_rhs[pp]",  Gt_src[2]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print("Building matter-source expressions (this loads dendrosym)...")
    assigns = build_matter_sources()
    print(f"  Built {len(assigns)} += assignments.")

    # Replace Derivatives and UndefFunctions with [pp]-flavored symbols.
    pp_assigns = []
    for lhs, rhs in assigns:
        pp_assigns.append((lhs, replace_derivs_and_funcs(sym.S(rhs))))

    # CSE the 10 expressions together so common subexpressions become temps.
    print("Running SymPy CSE on the 10 expressions together...")
    rhs_list = [r for _, r in pp_assigns]
    cse_subs, cse_rhss = sym.cse(rhs_list, symbols=sym.numbered_symbols("MAT_"))
    print(f"  CSE produced {len(cse_subs)} temporaries.")

    # Emit C++.
    lines = [
        "// Matter-source contributions to BSSN RHS components.",
        "// Generated by codegen/emda_matter_source.py -- do not edit.",
        "//",
        "// Overlays:",
        "//   trK_rhs       += 4*pi*alpha*(rho + perpTTrace)",
        "//   At_{ij}_rhs   += chi * (-8*pi*alpha) * trace_free(perpT)_{ij}",
        "//   CAP_Gt_i_rhs  += -16*pi*alpha/chi * stressCurrent[i]",
        "//",
        f"// Generated with {len(cse_subs)} CSE temporaries.",
        "{",
    ]
    from dendrosym.cascade.builder import expand_integer_pows
    for sym_name, sub_expr in cse_subs:
        code = sym.ccode(expand_integer_pows(sub_expr))
        lines.append(f"  const double {sym_name} = {code};")
    for (lhs, _), rhs_simplified in zip(pp_assigns, cse_rhss):
        code = sym.ccode(expand_integer_pows(rhs_simplified))
        lines.append(f"  {lhs} += {code};")
    lines.append("}")

    out = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(out)
    print(f"Wrote {args.output} ({len(out)} bytes, {len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
