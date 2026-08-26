"""
BSSN core variables .
"""

import argparse
import sys as sys

from dendrosym.cascade.systems.bssn.legacy import dendro
from sympy import symbols, sqrt, exp, Rational, Matrix

###################################################################
# Math Initialization **
###################################################################

l1, l2, l3, l4, eta = symbols("lambda[0] lambda[1] lambda[2] lambda[3] eta[pp]")
lf0, lf1 = symbols("lambda_f[0] lambda_f[1]")

# Additional parameters for damping term
R0 = symbols("BSSN_ETA_R0")
ep1, ep2 = symbols("BSSN_ETA_POWER[0] BSSN_ETA_POWER[1]")

xi1, xi2, xi3 = symbols("BSSN_XI[0] BSSN_XI[1] BSSN_XI[2] ")

# ------
# declare variables
a = dendro.scalar("alpha", "[pp]")
chi = dendro.scalar("chi", "[pp]")
K = dendro.scalar("K", "[pp]")

Gt = dendro.vec3("Gt", "[pp]")
b = dendro.vec3("beta", "[pp]")
B = dendro.vec3("B", "[pp]")

gt = dendro.sym_3x3("gt", "[pp]")
At = dendro.sym_3x3("At", "[pp]")

Gt_rhs = dendro.vec3("Gt_rhs", "[pp]")

# -----
# Lie derivative weight
weight = -Rational(2, 3)
weight_Gt = Rational(2, 3)

# specify the functions for computing first and second derivatives
dendro.d = lambda i, x: symbols("grad_%d_%s" % (i, x))
dendro.d2 = lambda i, j, x: symbols("grad2_%d_%d_%s" % (min(i, j), max(i, j), x))
dendro.ad = dendro.d
dendro.kod = dendro.undef

d = dendro.d
ad = dendro.ad
kod = dendro.kod
d2 = dendro.d2

# CAD/SSL SYMBOLS
t = symbols("t")  # time; needed for SSL
ham = symbols("ham[pp]")  # hamiltonian constraint violation
C_CAHD = symbols("BSSN_CAHD_C")  # coefficient for CAHD strength
dt = symbols("dt")  # simulation time step
dx_i = symbols("dx_i")  # spatial resolution of current grid
dx_min = symbols("dx_min")  # spatial resolution of finest grid

dendro.set_metric(gt)
igt = dendro.get_inverse_metric()

ham_temp_var = symbols("ham_temp")


eta_func = (
    R0
    * sqrt(sum([igt[i, j] * d(i, chi) * d(j, chi) for i, j in dendro.e_ij]))
    / ((1 - chi**ep1) ** ep2)
)


def compute_bssn_rhs(
    gauge="standard", eta_val=eta, enable_ssl=False, enable_cahd=False
):
    """
    Compute the RHS for the BSSN Equations
    """

    # precompute the christoffel symbols and ricci, note that this sets some global vars
    C1 = dendro.get_first_christoffel()
    C2 = dendro.get_second_christoffel()
    C2_spatial = dendro.get_complete_christoffel(chi)
    [R, Rt, Rphi, CalGt] = dendro.compute_ricci(Gt, chi)

    if enable_ssl:
        # enable slow-start lapse

        # W = chi ** Rational(1, 2)
        W = sqrt(chi)

        h = symbols("h_ssl")
        sig = symbols("sig_ssl")
        # h = 0.6
        # sig = 20
        a_rhs = (
            l1 * dendro.lie(b, a)
            - 2 * a * K
            - W * (h * exp(-(t**2) / (2 * sig**2))) * (a - W)
        )
    else:
        a_rhs = l1 * dendro.lie(b, a) - 2 * a * K

    # beta, the shift is gauge dependent
    if gauge == "rochester":
        b_rhs = [
            (
                xi2 * dendro.vec_j_ad_j(b, b[i])
                + Rational(3, 4) * xi3 * Gt[i]
                - eta_val * b[i]
            )
            for i in dendro.e_i
        ]
    else:
        # standard shift
        b_rhs = [
            (Rational(3, 4) * (lf0 + lf1 * a) * B[i] + l2 * dendro.vec_j_ad_j(b, b[i]))
            for i in dendro.e_i
        ]

    # metric evolution
    gt_rhs = dendro.lie(b, gt, weight) - 2 * a * At
    chi_rhs = dendro.lie(b, chi, weight) + Rational(2, 3) * (chi * a * K)

    if enable_cahd:
        # curvature-adjusted Hamiltonian constraint damping
        ham_computation = (
            sum(chi * igt[j, k] * R[j, k] for j, k in dendro.e_ij)
            - dendro.sqr(At)
            + Rational(2, 3) * K**2
        )
        # chi_rhs += C_CAHD * chi * (dt * dx_i / dx_min) * ham # Etienne's method
        chi_rhs += C_CAHD * chi * (dx_i**2 / dt) * ham_computation  # WKB's method

    # Now for the Extrinsic Curvature Evolution
    AikAkj = Matrix(
        [
            sum(
                [
                    At[i, k]
                    * sum([dendro.inv_metric[k, l] * At[l, j] for l in dendro.e_i])
                    for k in dendro.e_i
                ]
            )
            for i, j in dendro.e_ij
        ]
    )

    At_rhs = (
        dendro.lie(b, At, weight)
        + chi * dendro.trace_free(a * R - dendro.DiDj(a))
        + a * (K * At - 2 * AikAkj.reshape(3, 3))
    )

    K_rhs = (
        dendro.lie(b, K) - dendro.laplacian(a, chi) + a * (K * K / 3 + dendro.sqr(At))
    )

    # Conformal evolution (Gt)
    At_UU = dendro.up_up(At)

    Gt_rhs = (
        Matrix([sum(b[j] * ad(j, Gt[i]) for j in dendro.e_i) for i in dendro.e_i])
        - Matrix([sum(CalGt[j] * d(j, b[i]) for j in dendro.e_i) for i in dendro.e_i])
        + Rational(2, 3)
        * Matrix([CalGt[i] * sum(d(j, b[j]) for j in dendro.e_i) for i in dendro.e_i])
        + Matrix(
            [
                sum(
                    [
                        igt[j, k] * d2(j, k, b[i]) + igt[i, j] * d2(j, k, b[k]) / 3
                        for j, k in dendro.e_ij
                    ]
                )
                for i in dendro.e_i
            ]
        )
        - Matrix(
            [sum([2 * At_UU[i, j] * d(j, a) for j in dendro.e_i]) for i in dendro.e_i]
        )
        + Matrix(
            [
                sum([2 * a * dendro.C2[i, j, k] * At_UU[j, k] for j, k in dendro.e_ij])
                for i in dendro.e_i
            ]
        )
        - Matrix(
            [
                sum(
                    [
                        a
                        * (
                            3 / chi * At_UU[i, j] * d(j, chi)
                            + Rational(4, 3) * dendro.inv_metric[i, j] * d(j, K)
                        )
                        for j in dendro.e_i
                    ]
                )
                for i in dendro.e_i
            ]
        )
    )

    # flatten out Gt_rhs
    Gt_rhs = [item for sublist in Gt_rhs.tolist() for item in sublist]

    # B_rhs is based on Gt_rhs in standard gauge
    if gauge == "standard":
        B_rhs = [
            (
                Gt_rhs[i]
                - eta_val * B[i]
                + l3 * dendro.vec_j_ad_j(b, B[i])
                - l4 * dendro.vec_j_ad_j(b, Gt[i])
            )
            for i in dendro.e_i
        ]
    else:
        # rochester gauge doesn't evolve B
        B_rhs = [Rational(0, 1), Rational(0, 1), Rational(0, 1)]

    return {
        "a_rhs": a_rhs,
        "b_rhs": b_rhs,
        "gt_rhs": gt_rhs,
        "chi_rhs": chi_rhs,
        "At_rhs": At_rhs,
        "K_rhs": K_rhs,
        "Gt_rhs": Gt_rhs,
        "B_rhs": B_rhs,
        # NOTE: calgt is exposed for the staged structure
        "CalGt": CalGt,
    }


def generate_code(
    staged_type, gauge, eta_damp, prefix, enable_ssl, enable_cahd, generate_for_python,
    scheduler="dfs-outputs", dedup_threshold=2, precompute_igt=False,
):
    eta_val = eta_func if eta_damp == "func" else eta
    file_end = "py" if generate_for_python else "cpp"

    print(
        f"//Codgen: gauge={gauge}, eta={eta_damp}, ssl={enable_ssl}, cahd={enable_cahd}, python={generate_for_python}"
    )

    # optionally replace the symbolic inverse metric with opaque symbols
    # this keeps the expression trees small through christoffel/ricci computation
    igt_prefix_exprs = []
    igt_prefix_names = []
    if precompute_igt:
        print("//Codgen: precomputing inverse metric as opaque symbols")
        from sympy import Symbol, Matrix

        # save the real igt so we can grab its expressions
        real_igt = dendro.get_inverse_metric()

        # create placeholder symbols for the 6 independent components (symmetric)
        mi = [0, 1, 2, 4, 5, 8]  # indices into flattened 3x3
        midx = ["00", "01", "02", "11", "12", "22"]
        igt_syms = {}
        for j, k in enumerate(mi):
            # no [pp] suffix — these are local scalars, not array accesses
            sym = Symbol(f"igt{midx[j]}")
            row, col = k // 3, k % 3
            igt_syms[(row, col)] = sym
            igt_syms[(col, row)] = sym  # symmetric
            igt_prefix_exprs.append(real_igt[row, col])
            igt_prefix_names.append(f"igt{midx[j]}")

        # build the fake igt matrix from our placeholder symbols
        fake_igt = Matrix([
            [igt_syms[(0,0)], igt_syms[(0,1)], igt_syms[(0,2)]],
            [igt_syms[(1,0)], igt_syms[(1,1)], igt_syms[(1,2)]],
            [igt_syms[(2,0)], igt_syms[(2,1)], igt_syms[(2,2)]],
        ])

        # override the global inverse metric so all downstream math uses our symbols
        dendro.inv_metric = fake_igt

        # force christoffel recomputation with the new igt
        dendro.C1 = dendro.undef
        dendro.C2 = dendro.undef
        dendro.C3 = dendro.undef

    # compute everything
    rhs_dict = compute_bssn_rhs(gauge, eta_val, enable_ssl, enable_cahd)

    # extract the lists for code generation like we've done before
    outs = [
        rhs_dict["a_rhs"],
        rhs_dict["b_rhs"],
        rhs_dict["gt_rhs"],
        rhs_dict["chi_rhs"],
        rhs_dict["At_rhs"],
        rhs_dict["K_rhs"],
        rhs_dict["Gt_rhs"],
        rhs_dict["B_rhs"],
    ]
    vnames = [
        "a_rhs",
        "b_rhs",
        "gt_rhs",
        "chi_rhs",
        "At_rhs",
        "K_rhs",
        "Gt_rhs",
        "B_rhs",
    ]

    # then generate the code
    # TODO: staged generation
    if staged_type == "staged":
        # staging originally split stuff
        pass

    print("//Codgen: Generating Optimized Block Code...")

    flat_ex, flat_vnames, idx = dendro.construct_expression_list(outs, vnames, "[pp]")

    # if we precomputed igt, we don't need to modify flat_ex —
    # the igt symbols are already embedded in the expressions.
    # they'll be treated as input variables (like alpha, chi, etc.)
    # and the C++ wrapper will compute them before the RHS kernel.

    # cache the cse result so we don't redo 20 min of sympy work every run
    import pickle, hashlib, os
    cache_key = hashlib.md5(str(flat_ex).encode()).hexdigest()[:12]
    cache_file = f".cse_cache_{cache_key}.pkl"

    if os.path.exists(cache_file):
        print(f"//Codgen: loading cached CSE from {cache_file}")
        with open(cache_file, "rb") as f:
            cse_list = pickle.load(f)
    else:
        cse_list = dendro.construct_cse(flat_ex, flat_vnames, "[pp]")
        with open(cache_file, "wb") as f:
            pickle.dump(cse_list, f)
        print(f"//Codgen: saved CSE cache to {cache_file}")

    # post-processing: product dedup + scheduling (fast, runs every time)
    cse_list = dendro.optimize_cse(cse_list, scheduler=scheduler, dedup_threshold=dedup_threshold)

    output_code_original = dendro.generate_cpu_preextracted(
        cse_list[0],
        flat_vnames,
        "",
        cse_list[1],
        generate_for_python=generate_for_python,
    )

    with open(f"{prefix}_bssn_ORIGINAL.{file_end}", "w") as f:
        f.write(output_code_original)

    # output_code = dendro.generate_cpu_blocks(
    #     outs,
    #     vnames,
    #     "[pp]",
    #     cse_data=cse_list[0],
    #     orig_ops=cse_list[1],
    #     lname=vnames,
    #     lexp=ex,
    #     generate_for_python=generate_for_python,
    # )
    # with open(f"{prefix}_bssn_BLOCKS.{file_end}", "w") as f:
    #     f.write(output_code)

    # NOTE: block-based generation is slow, skip for now during iteration
    # output_code_inplace, output_code_graph = dendro.generate_cpu_blocks(
    #     flat_ex, flat_vnames, "[pp]",
    #     cse_data=cse_list[0], orig_ops=cse_list[1],
    #     generate_for_python=generate_for_python,
    #     return_inplace_and_non_inplace=True,
    #     use_register_aware_method=False,
    #     inplace_max_nodes_per_kernel=200, register_limit=25, skip_cse_calc=False,
    # )
    # with open(f"{prefix}_bssn_BLOCKS_INPLACE.{file_end}", "w") as f:
    #     f.write(output_code_inplace)
    # with open(f"{prefix}_bssn_BLOCKS_NOTINPLACE.{file_end}", "w") as f:
    #     f.write(output_code_graph)

    # optimized version: const-double ssa form with product dedup + ilp scheduling
    print("//Codgen: Generating Optimized (pow-expanded) Code...")
    output_code_fused = dendro.generate_cpu_preextracted(
        cse_list[0],
        flat_vnames,
        "",
        cse_list[1],
        generate_for_python=generate_for_python,
        use_const=True,
    )

    with open(f"{prefix}_bssn_FUSED.{file_end}", "w") as f:
        f.write(output_code_fused)

    # staged version: run cse independently per variable family
    # reduces register pressure at the cost of recomputing shared temps
    print("//Codgen: Generating Staged Code...")

    # group outputs by variable family (indices into flat_ex/flat_vnames)
    # ordering: a_rhs(1), b_rhs(3), gt_rhs(6), chi_rhs(1), At_rhs(6), K_rhs(1), Gt_rhs(3), B_rhs(3)
    stages = [
        ("simple", list(range(0, 11))),   # a_rhs + b_rhs + gt_rhs + chi_rhs
        ("K_rhs",  [17]),                  # K_rhs alone
        ("Gt_rhs", [18, 19, 20]),          # Gt_rhs
        ("B_rhs",  [21, 22, 23]),          # B_rhs
        ("At_rhs", [11, 12, 13, 14, 15, 16]),  # At_rhs (the bottleneck)
    ]

    staged_code = "// Dendro: STAGED GENERATION\n"
    total_staged_temps = 0
    total_staged_ops = 0

    for stage_name, indices in stages:
        stage_exprs = [flat_ex[i] for i in indices]
        stage_names = [flat_vnames[i] for i in indices]

        # run cse independently for this stage
        stage_cse = dendro.construct_cse(stage_exprs, stage_names, "[pp]")
        stage_cse = dendro.optimize_cse(stage_cse, scheduler=scheduler, dedup_threshold=dedup_threshold)

        stage_repl, stage_reduced = stage_cse[0]
        total_staged_temps += len(stage_repl)
        total_staged_ops += stage_cse[1]

        stage_code = dendro.generate_cpu_preextracted(
            stage_cse[0],
            stage_names,
            "",
            stage_cse[1],
            generate_for_python=generate_for_python,
            use_const=True,
            return_stats=False,
        )

        # wrap in braces so DENDRO_ names don't collide between stages
        staged_code += f"\n// --- stage: {stage_name} ({len(stage_repl)} temps, {len(indices)} outputs) ---\n"
        staged_code += "{\n"
        staged_code += stage_code
        staged_code += "}\n"

        # also emit each stage as its own file for separate-loop variant
        with open(f"{prefix}_bssn_STAGE_{stage_name}.{file_end}", "w") as f:
            f.write(stage_code)

    staged_code += f"\n// Dendro: STAGED TOTALS: {total_staged_temps} temps across {len(stages)} stages\n"

    with open(f"{prefix}_bssn_STAGED.{file_end}", "w") as f:
        f.write(staged_code)

    print(f"//Codgen: Staged generation complete: {total_staged_temps} total temps in {len(stages)} stages")

    # hybrid 2-stage: everything except At_rhs in one pass, At_rhs alone
    # minimizes recomputation while isolating the bottleneck
    print("//Codgen: Generating 2-Stage Hybrid Code...")
    hybrid_stages = [
        ("non_At", [i for i in range(24) if i not in range(11, 17)]),  # everything except At
        ("At_rhs", list(range(11, 17))),  # At_rhs alone
    ]

    hybrid_code = "// Dendro: 2-STAGE HYBRID GENERATION\n"
    total_hybrid_temps = 0

    for stage_name, indices in hybrid_stages:
        stage_exprs = [flat_ex[i] for i in indices]
        stage_names = [flat_vnames[i] for i in indices]

        stage_cse = dendro.construct_cse(stage_exprs, stage_names, "[pp]")
        stage_cse = dendro.optimize_cse(stage_cse, scheduler=scheduler, dedup_threshold=dedup_threshold)

        stage_repl, stage_reduced = stage_cse[0]
        total_hybrid_temps += len(stage_repl)

        stage_code = dendro.generate_cpu_preextracted(
            stage_cse[0], stage_names, "", stage_cse[1],
            generate_for_python=generate_for_python, use_const=True,
        )

        hybrid_code += f"\n// --- stage: {stage_name} ({len(stage_repl)} temps, {len(indices)} outputs) ---\n"
        hybrid_code += "{\n"
        hybrid_code += stage_code
        hybrid_code += "}\n"

    hybrid_code += f"\n// Dendro: HYBRID TOTALS: {total_hybrid_temps} temps across 2 stages\n"

    with open(f"{prefix}_bssn_HYBRID.{file_end}", "w") as f:
        f.write(hybrid_code)

    print(f"//Codgen: Hybrid generation complete: {total_hybrid_temps} total temps in 2 stages")


# choices...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="BSSN Code Generation",
        description="Generate the Code for the BSSN RHS equations.",
    )

    parser.add_argument(
        "-t",
        "--staged_type",
        choices=["staged", "unstaged"],
        default="unstaged",
        help="If we should use staged or unstaged code",
    )
    parser.add_argument(
        "-g",
        "--gauge",
        choices=["standard", "rochester"],
        default="standard",
        help="The gauge type",
    )
    parser.add_argument(
        "-e",
        "--eta_damp",
        choices=["const", "func"],
        default="const",
        help="The eta damping type, a function or a constant",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help="The file prefix output code",
        default="output_",
    )
    parser.add_argument(
        "-s",
        "--enable_ssl",
        action="store_true",
        help="Whether or not to generate with the SSL code",
    )
    parser.add_argument(
        "-c",
        "--enable_cahd",
        action="store_true",
        help="Whether or not to enable CAHD",
    )

    parser.add_argument(
        "-py",
        "--generate_for_python",
        action="store_true",
        help="If set, it generates for Python instead of C/C++",
    )
    parser.add_argument(
        "--scheduler",
        choices=["dfs-outputs", "critical-path", "min-live", "none"],
        default="dfs-outputs",
        help="CSE temp scheduling algorithm (default: dfs-outputs)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=int,
        default=2,
        dest="dedup_threshold",
        help="min occurrences for product dedup (default: 2, 0 to disable)",
    )
    parser.add_argument(
        "--precompute-igt",
        action="store_true",
        dest="precompute_igt",
        help="treat inverse metric components as opaque symbols to reduce expression tree depth",
    )

    args = parser.parse_args()

    generate_code(**vars(args))
