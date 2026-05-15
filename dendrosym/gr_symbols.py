"""gr_symbols.py

Pre-defined symbolic constants for GR solvers that map to Dendro
runtime values. Use these in symbolic initial data and analytical
solution expressions.

Usage:
    from dendrosym.gr_symbols import *

    # schwarzschild: alpha = (r - M/2) / (r + M/2)
    alpha_expr = (r - BH1_mass/2) / (r + BH1_mass/2)

    dendroConfigs.runtime_symbol_map = GR_RUNTIME_SYMBOLS
    dendroConfigs.initial_data_types = [{
        "id": 1, "name": "Schwarzschild",
        "function": "schwarzschildInit",
        "sympy_exprs": {alpha: alpha_expr, ...}
    }]
"""

import sympy as sym

# ---- spatial coordinates ----
x, y, z = sym.symbols("x y z")
r = sym.sqrt(x**2 + y**2 + z**2)
t = sym.Symbol("t")

# ---- BH1 parameters ----
BH1_mass = sym.Symbol("BH1_mass")
BH1_x = sym.Symbol("BH1_x")
BH1_y = sym.Symbol("BH1_y")
BH1_z = sym.Symbol("BH1_z")
BH1_spin = sym.Symbol("BH1_spin")

# ---- BH2 parameters ----
BH2_mass = sym.Symbol("BH2_mass")
BH2_x = sym.Symbol("BH2_x")
BH2_y = sym.Symbol("BH2_y")
BH2_z = sym.Symbol("BH2_z")
BH2_spin = sym.Symbol("BH2_spin")

# ---- TPID parameters ----
TPID_par_b = sym.Symbol("TPID_par_b")

# ---- mapping from sympy symbols to C++ runtime expressions ----
# pass this as dendroConfigs.runtime_symbol_map
GR_RUNTIME_SYMBOLS = {
    BH1_mass: sym.Symbol("BH1.getBHMass()"),
    BH1_x: sym.Symbol("BH1.getBHCoordX()"),
    BH1_y: sym.Symbol("BH1.getBHCoordY()"),
    BH1_z: sym.Symbol("BH1.getBHCoordZ()"),
    BH1_spin: sym.Symbol("BH1.getBHSpin()"),

    BH2_mass: sym.Symbol("BH2.getBHMass()"),
    BH2_x: sym.Symbol("BH2.getBHCoordX()"),
    BH2_y: sym.Symbol("BH2.getBHCoordY()"),
    BH2_z: sym.Symbol("BH2.getBHCoordZ()"),
    BH2_spin: sym.Symbol("BH2.getBHSpin()"),

    TPID_par_b: sym.Symbol("TPID::par_b"),
}
