"""mhd_cascade.py -- Drive CascadeBuilder + emit for ideal-MHD x-flux.

Mirrors neohook_cascade.py for the MHD spec in mhd.py. Same IR pipeline;
auto-generates scalar / AVX2 / AVX-512 / JAX kernels.

CLI:
    python mhd_cascade.py                                # natural depth
    python mhd_cascade.py --L 3                          # collapsed
    python mhd_cascade.py --L 7 --split-mode smart       # smart split
    python mhd_cascade.py --emit-cpp                     # scalar C++ body
    python mhd_cascade.py --emit-kernel --simd avx2      # AVX2 kernel function
"""

import argparse

from dendrosym.cascade.builder import build_cascade_ir
from dendrosym.cascade.systems.mhd import mhd_flux_spec


def build_ir(target_L=None, per_point=False, smart_split=None, verbose=False):
    """Build the MHD-flux cascade with optional collapse/split to target_L."""
    chunks, leaves = mhd_flux_spec(per_point=per_point)
    return build_cascade_ir(chunks, leaves, target_L=target_L,
                            smart_split=smart_split, verbose=verbose)


def emit_cpp(target_L=None, verbose=False):
    """Scalar body, bare-name leaves (for a per-iteration locals wrapper)."""
    result = build_ir(target_L=target_L, per_point=False, verbose=verbose)
    L = len(result.chunks)
    body = result.emit_cpp_unrolled(
        dendro_var_style=False, inline_threshold=0, short_names=False
    )
    header = (
        "// Ideal MHD x-flux via polynomial cascade -- IR-driven.\n"
        f"// L = {L} chunks; per-chunk CSE only.\n"
        "// Inputs: rho, mom_x, mom_y, mom_z, E, Bx, By, Bz, gamma_minus_1.\n"
        "// Outputs: F_rho, F_mom_{x,y,z}, F_E, F_{Bx,By,Bz}.\n"
        "\n"
    )
    return header + body + "\n"


def emit_kernel(target_L=None, simd="avx2", fold_vfma=0, smart_split=None,
                verbose=False, fn_name="mhd_flux_kernel"):
    """Auto-generate a complete C++ kernel function for MHD flux."""
    from dendrosym.cascade.emit import emit_kernel_function_cpp
    result = build_ir(target_L=target_L, per_point=True,
                      smart_split=smart_split, verbose=verbose)
    return emit_kernel_function_cpp(
        result, fn_name=fn_name, simd=simd, fold_vfma_passes=fold_vfma,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--L", type=int, default=None,
                    help="target cascade depth (default: natural 5)")
    ap.add_argument("--emit-cpp", action="store_true",
                    help="print scalar body for the bare-name (locals) wrapper")
    ap.add_argument("--emit-kernel", action="store_true",
                    help="emit a complete auto-generated kernel function")
    ap.add_argument("--simd", choices=("scalar", "avx2", "avx512"),
                    default="avx2",
                    help="SIMD dialect when emitting a kernel function")
    ap.add_argument("--vfma", type=int, default=0,
                    help="VFMA folding passes (only with simd=avx2/avx512)")
    ap.add_argument("--split-mode", choices=("auto", "smart", "dumb"),
                    default="auto",
                    help="split strategy when L > natural (default auto)")
    ap.add_argument("--fn-name", default="mhd_flux_kernel")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    smart = {"auto": None, "smart": True, "dumb": False}[args.split_mode]
    if args.emit_kernel:
        print(emit_kernel(target_L=args.L, simd=args.simd,
                          fold_vfma=args.vfma,
                          smart_split=smart,
                          fn_name=args.fn_name,
                          verbose=args.verbose), end="")
    elif args.emit_cpp:
        print(emit_cpp(target_L=args.L, verbose=args.verbose), end="")
    else:
        result = build_ir(target_L=args.L,
                          smart_split=smart,
                          verbose=args.verbose)
        result.summary()


if __name__ == "__main__":
    main()
