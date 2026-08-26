"""neohook_cascade.py -- Drive CascadeBuilder + emit for 2D Neo-Hookean.

Mirrors bssn_cascade.build_ir() but for the talk's canonical Neo-Hookean
example. Demonstrates that the cascade pipeline isn't BSSN-shaped:
    neohook.neohook_2d_spec()  →  CascadeBuilder.add_chunk + .build()
       →  CascadeResult  →  cascade_emit (scalar)  →  C++ body
The same path can apply collapse/split for L != natural.

CLI:
    python neohook_cascade.py                 # natural depth (5)
    python neohook_cascade.py --L 3            # collapsed
    python neohook_cascade.py --L 7            # split
    python neohook_cascade.py --emit-cpp       # print scalar C++ body
"""

import argparse

from dendrosym.cascade.builder import build_cascade_ir
from dendrosym.cascade.systems.neohook import neohook_2d_spec, neohook_3d_spec


def build_ir(target_L=None, per_point=False, smart_split=None,
             dim=2, verbose=False):
    """Build the Neo-Hookean cascade with optional collapse/split to target_L.

    `dim=2` (default) uses neohook_2d_spec; `dim=3` uses neohook_3d_spec.
    See cascade_builder.build_cascade_ir for target_L / smart_split semantics.
    """
    spec_fn = neohook_3d_spec if dim == 3 else neohook_2d_spec
    chunks, leaves = spec_fn(per_point=per_point)
    return build_cascade_ir(chunks, leaves, target_L=target_L,
                            smart_split=smart_split, verbose=verbose)


def emit_cpp(target_L=None, verbose=False):
    """Scalar body, bare-name leaves (the original Phase 3 standalone bench)."""
    result = build_ir(target_L=target_L, per_point=False, verbose=verbose)
    L = len(result.chunks)
    body = result.emit_cpp_unrolled(
        dendro_var_style=False, inline_threshold=0, short_names=False
    )
    header = (
        "// Neo-Hookean 2D stress P_ij via polynomial cascade -- IR-driven.\n"
        f"// L = {L} chunks; per-chunk CSE only.\n"
        "// Inputs: a, b, c, d (∇u entries), mu, lam (material params).\n"
        "// Outputs: P00, P01, P10, P11.\n"
        "\n"
    )
    return header + body + "\n"


def emit_kernel(target_L=None, simd="avx2", fold_vfma=0, smart_split=None,
                dim=2, verbose=False, fn_name=None):
    """Auto-generate a complete C++ kernel function for Neo-Hookean."""
    from dendrosym.cascade.emit import emit_kernel_function_cpp
    if fn_name is None:
        fn_name = f"neohook{dim}d_kernel"
    result = build_ir(target_L=target_L, per_point=True, dim=dim,
                      smart_split=smart_split, verbose=verbose)
    return emit_kernel_function_cpp(
        result, fn_name=fn_name, simd=simd, fold_vfma_passes=fold_vfma,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--L", type=int, default=None,
                    help="target cascade depth (default: natural 5)")
    ap.add_argument("--emit-cpp", action="store_true",
                    help="print the scalar C++ body for the original bench")
    ap.add_argument("--emit-kernel", action="store_true",
                    help="emit a complete auto-generated kernel function")
    ap.add_argument("--simd", choices=("scalar", "avx2", "avx512"),
                    default="avx2",
                    help="SIMD dialect when emitting a kernel function")
    ap.add_argument("--vfma", type=int, default=0,
                    help="VFMA folding passes (only with simd=avx2/avx512)")
    ap.add_argument("--split-mode", choices=("auto", "smart", "dumb"),
                    default="auto",
                    help="split strategy when L > natural. 'auto' uses smart "
                         "for even-delta L, dumb otherwise")
    ap.add_argument("--dim", type=int, default=2, choices=(2, 3),
                    help="Spatial dimension (2 or 3). Default 2.")
    ap.add_argument("--fn-name", default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    smart = {"auto": None, "smart": True, "dumb": False}[args.split_mode]
    if args.emit_kernel:
        print(emit_kernel(target_L=args.L, simd=args.simd,
                          fold_vfma=args.vfma,
                          smart_split=smart, dim=args.dim,
                          fn_name=args.fn_name,
                          verbose=args.verbose), end="")
    elif args.emit_cpp:
        print(emit_cpp(target_L=args.L, verbose=args.verbose), end="")
    else:
        result = build_ir(target_L=args.L,
                          smart_split=smart, dim=args.dim,
                          verbose=args.verbose)
        result.summary()


if __name__ == "__main__":
    main()
