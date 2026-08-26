"""bssn_cascade.py -- IR-driven BSSN cascade construction.

Drives CascadeBuilder from the BSSN physics in bssn_physics.compute_physics
(the single source of truth), keeping named tensor intermediates (igt, C1, C2,
C3, R, At_UU, ...) as the chunk boundaries. CSE runs only inside each chunk.

Physics was extracted to bssn_physics so this module and bssn_clean /
bssn_decorated share ONE copy of the equations. Each driver keeps its own
spec-assembly code, so the bssn_clean self-test cross-checks the two
assemblies (the regression lock). The xRphi-correct Ricci that works around
dendro.compute_ricci's regression also lives in bssn_physics.
"""

import sympy as sym
from collections import OrderedDict

from dendrosym.cascade.builder import (
    build_cascade_ir, flatten_sym33, flatten_tensor3, flatten_vec3, E_IJ_SYM,
)
from dendrosym.cascade.systems.bssn.physics import compute_physics, RHS_OUTPUT_NAMES


def build_specs(gauge="standard", ssl=False, cahd=False, eta_mode="scalar"):
    """Construct the BSSN cascade as an ordered list of (name, outputs_dict)
    chunk specs and the leaf-symbol set, *without* running per-chunk CSE.

    Physics from bssn_physics.compute_physics; gauge / ssl / cahd / eta_mode are
    forwarded there. This module keeps its own flatten-based spec assembly so
    the bssn_clean self-test cross-checks both assemblies.
    """
    p = compute_physics(gauge=gauge, ssl=ssl, cahd=cahd, eta_mode=eta_mode)

    specs = []

    # L1: inverse metric + 1/chi
    igt_outputs = flatten_sym33(p.igt, "igt")
    igt_outputs["chi_inv"] = 1 / p.chi
    specs.append(("inverse_metric", igt_outputs))

    # L2: first Christoffel (symmetric in the last two indices)
    c1_outputs = OrderedDict()
    for k in range(3):
        for i in range(3):
            for j in range(i, 3):
                c1_outputs[f"C1_{k}{i}{j}"] = p.C1[k, i, j]
    specs.append(("first_christoffel", c1_outputs))

    # L3, L4: second + complete Christoffel
    specs.append(("second_christoffel", flatten_tensor3(p.C2, "C2_")))
    specs.append(("complete_christoffel", flatten_tensor3(p.C3, "C3_")))

    # L5: Ricci (corrected R = Rt + Rphi + xRphi) + CalGt
    ricci_outputs = flatten_sym33(p.R, "R")
    ricci_outputs.update(flatten_vec3(p.CalGt, "CalGt"))
    specs.append(("ricci", ricci_outputs))

    # L6: derived quantities
    derived = OrderedDict()
    derived.update(flatten_sym33(p.At_UU, "At_UU"))
    derived.update(flatten_sym33(p.AikAkj, "AikAkj"))
    derived.update(flatten_sym33(p.DiDj_a, "DiDj_a"))
    derived.update(flatten_sym33(p.tf, "tf"))
    derived["At_sqr"] = p.At_sqr
    derived["lap_a"] = p.lap_a
    specs.append(("derived_quantities", derived))

    # L7: RHS assembly
    rhs_outputs = OrderedDict()
    rhs_outputs["a_rhs"] = p.a_rhs
    for i in range(3):
        rhs_outputs[f"b_rhs{i}"] = p.b_rhs[i]
    for i, j in E_IJ_SYM:
        rhs_outputs[f"gt_rhs{i}{j}"] = p.gt_rhs[i, j]
    rhs_outputs["chi_rhs"] = p.chi_rhs
    for i, j in E_IJ_SYM:
        rhs_outputs[f"At_rhs{i}{j}"] = p.At_rhs[i, j]
    rhs_outputs["K_rhs"] = p.K_rhs
    for i in range(3):
        rhs_outputs[f"Gt_rhs{i}"] = p.Gt_rhs_list[i]
    for i in range(3):
        rhs_outputs[f"B_rhs{i}"] = p.B_rhs[i]
    specs.append(("rhs_assembly", rhs_outputs))

    assert frozenset(rhs_outputs.keys()) == RHS_OUTPUT_NAMES, (
        "rhs_assembly keys drifted from RHS_OUTPUT_NAMES; mismatch: "
        f"{frozenset(rhs_outputs.keys()) ^ RHS_OUTPUT_NAMES}")

    # leaves = free symbols of the final RHS not produced by any chunk
    leaves = set()
    for e in p.all_rhs:
        leaves |= e.free_symbols
    leaves -= {sym.Symbol(k) for _n, outs in specs for k in outs.keys()}
    return specs, leaves


def build_ir(dendro_module=None, bssn_module=None, target_L=None,
             smart_split=None, gauge="standard", ssl=False, cahd=False,
             eta_mode="scalar", verbose=False,
             auto_layers=False, auto_search_order=False):
    """Build the BSSN cascade (corrected Ricci). See build_specs for gauge opts;
    target_L / smart_split control collapse/split depth (see build_cascade_ir)."""
    specs, leaves = build_specs(gauge=gauge, ssl=ssl, cahd=cahd, eta_mode=eta_mode)
    return build_cascade_ir(specs, leaves, target_L=target_L,
                            smart_split=smart_split, verbose=verbose,
                            auto_layers=auto_layers,
                            auto_search_order=auto_search_order)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=None,
                   help="target cascade depth (default: physics-natural 7)")
    p.add_argument("--split-mode", choices=("auto", "smart", "dumb"),
                   default="auto",
                   help="split strategy when L > natural")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    smart = {"auto": None, "smart": True, "dumb": False}[args.split_mode]
    result = build_ir(target_L=args.L, smart_split=smart, verbose=args.verbose)
    result.summary()
