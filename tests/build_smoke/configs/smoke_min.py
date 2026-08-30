"""Minimum pole of the build smoke matrix: every feature flag off.

Two scalars and a Laplacian. Checks that a solver with no GR capabilities
still generates and compiles. This is what caught `enable_ah` defaulting to
True, which had a scalar-wave solver linking BHaHAHA.

Separate from the repo-root wave config, which changes for physics reasons.
"""

import dendrosym

dendroConfigs = dendrosym.NRConfig("smokemin")
idx_str = "[pp]"
dendroConfigs.set_idx_str(idx_str)

d_ = dendrosym.nr.set_first_derivative("grad")
d2_ = dendrosym.nr.set_second_derivative("grad2")
ad_ = dendrosym.nr.set_advective_derivative("agrad")
kod_ = dendrosym.nr.set_kreiss_oliger_dissipation("kograd")

u = dendrosym.dtypes.scalar("u" + idx_str)
v = dendrosym.dtypes.scalar("v" + idx_str)
dendroConfigs.add_evolution_variables([u, v])


def evolution_rhs_eqns():
    return [v, sum(d2_(i, i, u) for i in range(3))], [u, v]


dendroConfigs.set_rhs_equation_function("evolution", evolution_rhs_eqns)
dendroConfigs.set_bhs_falloff_and_asymptotic(
    "evolution", [u, v], [[1.0, 0.0], [1.0, 0.0]]
)
dendroConfigs.deriv_obj = "SOLVER_DERIVS"

if __name__ == "__main__":
    dendrosym.run(dendroConfigs, default_output_dir="./output/smokemin-solver")
