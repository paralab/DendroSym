"""em4_cascade.py -- Drive CascadeBuilder + emit for the EM4 (Maxwell+GLM) RHS.

Mirrors mhd_cascade.py. EM4 is a first-order system with no shared
intermediates, so the cascade is a single chunk (natural L=1); the value here
is the SIMD emitter on a first-order system, not the layering.

CLI:
    python em4_cascade.py                              # summary
    python em4_cascade.py --emit-cpp                   # scalar C++ body
    python em4_cascade.py --emit-kernel --simd avx2    # AVX2 kernel function
    python em4_cascade.py --emit-kernel --simd avx512  # AVX-512 kernel function
"""
import argparse

from dendrosym.cascade.builder import build_cascade_ir
from dendrosym.cascade.systems.em4 import em4_rhs_spec


def build_ir(target_L=None, per_point=False, smart_split=None, verbose=False):
    chunks, leaves = em4_rhs_spec(per_point=per_point)
    return build_cascade_ir(chunks, leaves, target_L=target_L,
                            smart_split=smart_split, verbose=verbose)


def emit_cpp(target_L=None, verbose=False):
    result = build_ir(target_L=target_L, per_point=False, verbose=verbose)
    L = len(result.chunks)
    body = result.emit_cpp_unrolled(
        dendro_var_style=False, inline_threshold=0, short_names=False
    )
    header = (
        "// EM4 (Maxwell + GLM cleaning) RHS via polynomial cascade -- IR-driven.\n"
        f"// L = {L} chunk(s); first-order system, no shared intermediates.\n"
        "// Inputs: grad_{0,1,2}_{E0..2,B0..2,Phi,Psi}, Phi, Psi, J0..2, rho_e,\n"
        "//         kappa_1, kappa_2.  Outputs: B_rhs0..2, E_rhs0..2, Phi_rhs, Psi_rhs.\n\n"
    )
    return header + body + "\n"


def emit_kernel(target_L=None, simd="avx2", fold_vfma=0, smart_split=None,
                verbose=False, fn_name="em4_rhs_kernel"):
    from dendrosym.cascade.emit import emit_kernel_function_cpp
    result = build_ir(target_L=target_L, per_point=True,
                      smart_split=smart_split, verbose=verbose)
    return emit_kernel_function_cpp(
        result, fn_name=fn_name, simd=simd, fold_vfma_passes=fold_vfma,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--emit-cpp", action="store_true")
    ap.add_argument("--emit-kernel", action="store_true")
    ap.add_argument("--simd", choices=("scalar", "avx2", "avx512"),
                    default="avx2")
    ap.add_argument("--vfma", type=int, default=0)
    ap.add_argument("--split-mode", choices=("auto", "smart", "dumb"),
                    default="auto")
    ap.add_argument("--fn-name", default="em4_rhs_kernel")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    smart = {"auto": None, "smart": True, "dumb": False}[args.split_mode]
    if args.emit_kernel:
        print(emit_kernel(target_L=args.L, simd=args.simd, fold_vfma=args.vfma,
                          smart_split=smart, fn_name=args.fn_name,
                          verbose=args.verbose), end="")
    elif args.emit_cpp:
        print(emit_cpp(target_L=args.L, verbose=args.verbose), end="")
    else:
        result = build_ir(target_L=args.L, smart_split=smart,
                          verbose=args.verbose)
        result.summary()


if __name__ == "__main__":
    main()
