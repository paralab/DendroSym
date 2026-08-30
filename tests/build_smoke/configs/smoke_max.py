"""Maximum pole of the build smoke matrix: every feature flag on.

A BSSN-shaped variable set with trivial equations. Compiles the guarded code
the render-only template gate cannot reach: TwoPunctures wiring, puncture
tracker, Psi4 extraction, BHaHAHA, analytic comparison, profiling timers.

The variable names matter even though the equations do not: the AH input
mapping resolves against `chi/trK/At[6]/gt[6]`, and GW extraction reads
`psi4_real/psi4_imag`.
"""

import dendrosym
import sympy as sym

dendroConfigs = dendrosym.NRConfig("smokemax")
idx_str = "[pp]"
dendroConfigs.set_idx_str(idx_str)

d_ = dendrosym.nr.set_first_derivative("grad")
d2_ = dendrosym.nr.set_second_derivative("grad2")
ad_ = dendrosym.nr.set_advective_derivative("agrad")
kod_ = dendrosym.nr.set_kreiss_oliger_dissipation("kograd")

# ---- evolution variables (BSSN names; see the module docstring) ----
alpha = dendrosym.dtypes.scalar("alpha" + idx_str)
chi = dendrosym.dtypes.scalar("chi" + idx_str)
trK = dendrosym.dtypes.scalar("trK" + idx_str)
beta = dendrosym.dtypes.vec3("beta" + idx_str)
Gt = dendrosym.dtypes.vec3("Gt" + idx_str)
B = dendrosym.dtypes.vec3("B" + idx_str)
At = dendrosym.dtypes.sym_3x3("At" + idx_str)
gt = dendrosym.dtypes.sym_3x3("gt" + idx_str)

dendroConfigs.add_evolution_variables([alpha, chi, trK, beta, Gt, B, At, gt])
dendroConfigs.set_metric(gt)

# ---- constraint variables (psi4_* feed the GW extraction) ----
ham = dendrosym.dtypes.scalar("ham" + idx_str)
mom = dendrosym.dtypes.vec3("mom" + idx_str)
psi4_real = dendrosym.dtypes.scalar("psi4_real" + idx_str)
psi4_imag = dendrosym.dtypes.scalar("psi4_imag" + idx_str)
dendroConfigs.add_constraint_variables([ham, mom, psi4_real, psi4_imag])


def _lap(x):
    return sum(d2_(i, i, x) for i in range(3))


def evolution_rhs_eqns():
    scalars = [alpha, chi, trK]
    vectors = [beta, Gt, B]
    rhs = [_lap(s) for s in scalars]
    rhs += [[_lap(v[i]) for i in range(3)] for v in vectors]
    # rank-2 rhs must be a sympy Matrix, not nested lists
    rhs += [dendrosym.nr.rank2(lambda i, j, m=m: _lap(m[i, j]))
            for m in (At, gt)]
    return rhs, [alpha, chi, trK, beta, Gt, B, At, gt]


def constraint_rhs_eqns():
    return (
        [_lap(chi), [d_(i, chi) for i in range(3)], _lap(alpha), _lap(trK)],
        [ham, mom, psi4_real, psi4_imag],
    )


dendroConfigs.set_rhs_equation_function("evolution", evolution_rhs_eqns)
dendroConfigs.set_rhs_equation_function("constraint", constraint_rhs_eqns)

_fa = [1.0, 0.0]
dendroConfigs.set_bhs_falloff_and_asymptotic(
    "evolution",
    [alpha, chi, trK, beta, Gt, B, At, gt],
    [_fa, _fa, _fa, _fa, _fa, _fa, _fa, _fa],
)

# pointwise enforcement -- exercises the evolution-constraint path
dendroConfigs.add_evolution_constraint(At, "trace_zero")
dendroConfigs.add_evolution_constraint(chi, "pos_floor")

# ---- initial data + an analytic solution to compare against ----
x_s, y_s, z_s, t_s = sym.symbols("x y z t")
r2 = x_s**2 + y_s**2 + z_s**2
_flat = {alpha: sym.Integer(1), chi: sym.Integer(1), trK: sym.Integer(0)}
_flat.update({gt[i, i]: sym.Integer(1) for i in range(3)})
dendroConfigs.symbolic_initial_data = dict(_flat)
dendroConfigs.symbolic_initial_data_name = "smokeFlatInit"
dendroConfigs.symbolic_analytical_solution = {
    alpha: sym.Integer(1) + sym.Rational(1, 100) * sym.exp(-r2) * sym.cos(t_s),
}

# ---- every feature flag on ----
dendroConfigs.enable_bh_tracking = True
dendroConfigs.enable_gw_extraction = True
dendroConfigs.enable_ah = True
dendroConfigs.enable_analytical = True
dendroConfigs.enable_profiling = True
dendroConfigs.enable_tpid = True
# The writer stub is generated on first run and then owned by the author; here
# it is exactly what we want to compile, since a from-scratch TPID solver has to
# be buildable without anyone hand-writing a file they were never told about.
dendroConfigs.tpid_writer = "smokemax::writePunctureVars"

dendroConfigs.deriv_obj = "SOLVER_DERIVS"

if __name__ == "__main__":
    dendrosym.run(dendroConfigs, default_output_dir="./output/smokemax-solver")
