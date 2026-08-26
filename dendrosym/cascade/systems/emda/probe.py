"""emdabssn.py -- Phase-1 inspection of the EMDA-BSSN equation system.

Loads the SymPy/dendrosym configuration from ~/research/emda-gr and reports:

  - state-variable count (expected: 8 BSSN + 12 matter = 20)
  - RHS-expression count
  - 1st- and 2nd-derivative footprint per state variable
  - rough symbolic-size summary (terms per RHS, total ops)

This is the smallest "does dendrosym work end-to-end here" probe we need before
extending cascade_codegen.py to a second PDE system. No code is emitted.

Usage:
    PYTHONPATH=~/research/emda-gr python emdabssn.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter

# Add emda-gr to sys.path so emdabssn_eqns_configs is importable.
EMDA_GR_PATH = os.environ.get("DENDRO_EMDA_GR", os.path.expanduser("~/research/emda-gr"))
if EMDA_GR_PATH not in sys.path:
    sys.path.insert(0, EMDA_GR_PATH)

import sympy as sym  # noqa: E402

import emdabssn_eqns_configs as E  # noqa: E402


# dendrosym holds derivatives as real SymPy Derivative objects with the
# field as an UndefinedFunction over the coordinate symbols (xx_temp, yy_temp,
# zz_temp). Map those to direction indices 0/1/2.
_COORD_DIR = {"xx_temp": 0, "yy_temp": 1, "zz_temp": 2}


def deriv_footprint(expr: sym.Expr) -> tuple[set, set]:
    """Walk Derivative atoms and bucket as (dir, field) for 1st and
    (d1, d2, field) (sorted) for 2nd derivatives. EMDA's get_rhs_eqns
    output uses no advective shorthand -- advection appears as plain
    Derivative wrt xx/yy/zz_temp with an explicit beta multiplier.
    """
    g1, g2 = set(), set()
    for d in expr.atoms(sym.Derivative):
        # d.expr is an UndefinedFunction call like gt00(xx_temp, yy_temp, zz_temp)
        fld = d.expr.func.__name__
        # d.variables is a tuple of coord symbols. Each appears once for each
        # order of differentiation.
        dirs = []
        for v in d.variables:
            name = str(v)
            if name not in _COORD_DIR:
                return g1, g2  # foreign coord — bail
            dirs.append(_COORD_DIR[name])
        if len(dirs) == 1:
            g1.add((dirs[0], fld))
        elif len(dirs) == 2:
            i, j = sorted(dirs)
            g2.add((i, j, fld))
        # higher-order: ignore (BSSN/EMDA never go beyond 2nd)
    return g1, g2


def main() -> None:
    print(f"[emdabssn] dendrosym at {sys.modules['dendrosym'].__file__}")
    print(f"[emdabssn] config: {E.dendroConfigs}")

    rhs_exprs, rhs_names = E.dendroConfigs.get_rhs_eqns_flat("evolution")
    print(f"\n=== EVOLUTION ({len(rhs_exprs)} flat RHS expressions) ===")

    # Per-RHS summary
    deriv1_global, deriv2_global = set(), set()
    fields_used: Counter[str] = Counter()
    sizes = []
    for name, expr in zip(rhs_names, rhs_exprs):
        e = sym.S(expr)
        nops = sym.count_ops(e, visual=False)
        nterms = len(e.args) if e.is_Add else 1
        g1, g2 = deriv_footprint(e)
        deriv1_global |= g1
        deriv2_global |= g2
        for _, fld in g1:
            fields_used[fld] += 1
        for _, _, fld in g2:
            fields_used[fld] += 1
        sizes.append((str(name), nops, nterms, len(g1), len(g2)))

    print(f"{'RHS':<24} {'ops':>7} {'terms':>6} {'d1':>4} {'d2':>4}")
    for name, nops, nterms, n1, n2 in sizes:
        print(f"{name:<24} {nops:>7} {nterms:>6} {n1:>4} {n2:>4}")

    print(f"\nTotals: ops={sum(s[1] for s in sizes):>8}, "
          f"unique 1st-derivs={len(deriv1_global)}, "
          f"unique 2nd-derivs={len(deriv2_global)}")

    n2_pure = sum(1 for i, j, _ in deriv2_global if i == j)
    n2_mixed = sum(1 for i, j, _ in deriv2_global if i != j)
    print(f"  pure 2nd derivs (xx/yy/zz): {n2_pure}, mixed (xy/xz/yz): {n2_mixed}")

    fields_with_derivs = sorted(fields_used.keys())
    print(f"\nFields appearing under d/d2 ({len(fields_with_derivs)}):")
    for f in fields_with_derivs:
        print(f"  {f:<20} (referenced by {fields_used[f]} RHS exprs)")


if __name__ == "__main__":
    main()
