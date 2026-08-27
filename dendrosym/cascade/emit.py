"""cascade_emit.py -- IR-to-C++ emitter for the cascade pipeline.

Takes a CascadeResult (from cascade_builder.CascadeBuilder.build()) and emits
a per-point C++ body suitable for #include into the bssnrhs harness loop.

BEST / DEPLOYED config: `--simd avx512 --L 7 --ssl --cahd --fused` (fma-tree is
already default). This is the paper's headline variant. The defaults below are
vacuum + scalar on purpose -- the clean reproducibility baseline / research
control -- so the deployed config is explicit opt-in, not the default.

Phase 1a: scalar-only output. The body uses plain C++ arithmetic; the harness
wrapper compiles it in a `for (i,j,k)` loop with `#pragma omp simd`.

Phase 1b: AVX2 output. The body declares VEC variables, uses VMUL/VADD/VFMA
macros, runs inside a 4-wide batched outer loop in a separate harness wrapper
(bssn_cascade_ir_avx2_l_rhs.cpp). Same source compiles under SCALAR/AVX2/
AVX-512 macro headers from cascade_common.py.

Emitter overview and when-to-use table: findings/cascade_api_guide.md §3.

CLI:
    python cascade_emit.py --output ../harness/src/rhsfuncs/generated/bssneqs_cascade_ir.cpp
    python cascade_emit.py --simd avx2 --L 7 --output ../harness/src/rhsfuncs/generated/bssneqs_cascade_ir_avx2_L7.cpp
"""

import argparse
import os
import sys
import re

import sympy as sym


def _inline_low_use_temps(result, threshold: int = 2):
    """Post-build pass: inline CSE temps used ≤ threshold times.

    SymPy's per-chunk CSE creates a named temp for any subexpression appearing
    ≥ 2 times. That's optimal for code size but creates many short-lived
    declarations the compiler has to keep track of. The legacy hand-written
    emit_L1..emit_L7 in cascade_codegen.py uses far fewer named temps
    (~92 named scalars total for BSSN vs IR's ~870), and benchmarks show
    the legacy form is ~3× faster at the assembly-instruction level.

    This pass counts each temp's references across all downstream chunks
    (CSE temps + outputs). Temps with ≤ threshold uses get xreplace'd into
    their consumers and dropped. Outputs and chunk-boundary names are
    preserved (those are the named tensors the cascade is built around).

    Returns a new CascadeResult.
    """
    from dendrosym.cascade.builder import ChunkResult, CascadeResult
    from collections import OrderedDict

    # Build global symbol → expression map from all CSE temps across all chunks.
    # Also note which symbols are "boundary outputs" (chunk outputs) — we
    # never inline those; they're the cascade structure.
    chunk_outputs = set()
    for c in result.chunks:
        chunk_outputs |= set(c.outputs.keys())

    # Build inlining-by-chunk: each CSE temp lives within its chunk and may
    # be referenced by other CSE temps in that chunk, the chunk's outputs,
    # and (via output Symbols) by downstream chunks. But SymPy CSE temps
    # have unique CASC_<chunk_name>_N names so they only appear in their
    # own chunk's expressions.
    # Count refs by string-counting in serialized expressions per chunk.

    new_chunks = []
    for c in result.chunks:
        # Collect all expressions in this chunk: cse_temps RHS and output exprs.
        # Count free-symbol occurrences for each CSE temp's defined symbol.
        temp_symbols = [s for s, _ in c.cse_temps]
        temp_set = set(s.name for s in temp_symbols)
        if not temp_symbols:
            new_chunks.append(c)
            continue

        ref_counts = {n: 0 for n in temp_set}
        # Count uses across all later temps' RHS and all outputs' RHS.
        for s2, e2 in c.cse_temps:
            for fs in e2.free_symbols:
                if fs.name in ref_counts and fs.name != s2.name:
                    ref_counts[fs.name] += 1
        for _, eout in c.outputs.items():
            for fs in eout.free_symbols:
                if fs.name in ref_counts:
                    ref_counts[fs.name] += 1

        # Determine which to inline (used ≤ threshold times).
        # Inline lowest-ref-count first so chained inlines compound.
        order = sorted(temp_symbols, key=lambda s: ref_counts.get(s.name, 0))
        rhs = {s.name: e for s, e in c.cse_temps}
        inlined = set()
        # We process in original CSE order so each inlined temp's RHS doesn't
        # contain references to later temps. Build subs progressively.
        subs = {}
        new_temps = []
        for s, e in c.cse_temps:
            # Apply existing subs to this temp's RHS so subsequent inlines
            # have flat expressions.
            e_subbed = e.xreplace(subs) if subs else e
            n_uses = ref_counts.get(s.name, 0)
            if n_uses <= threshold:
                subs[s] = e_subbed
                inlined.add(s.name)
            else:
                new_temps.append((s, e_subbed))

        # Apply final subs to outputs.
        new_outputs = OrderedDict()
        for name, e in c.outputs.items():
            new_outputs[name] = e.xreplace(subs) if subs else e

        new_chunks.append(ChunkResult(
            name=c.name,
            cse_temps=new_temps,
            outputs=new_outputs,
            input_symbols=c.input_symbols,
            n_temps=len(new_temps),
        ))

    return CascadeResult(chunks=new_chunks, leaf_symbols=result.leaf_symbols)


def _classify_leaves(result, fused=False):
    """Walk a CascadeResult and bucket leaf symbols.

    Returns a dict:
        pp_arrays  : list[str]  array names that get VLOAD'd at offset pp
        idx_consts : list[(name, idx)] e.g. ('lambda', 0)..('lambda_f', 1)
        scalars    : list[str]  scalar names already in VEC scope (eta)
        stencil_1st: list[(arr, d, var)]  (fused only) grad_d_var / agrad_d_var
                     leaves to compute via inline 6th-order stencils
        stencil_2nd: list[(arr, d, var)]  (fused only) pure grad2_d_d_var
    plus a `rename_map: dict[Symbol, Symbol]` to xreplace into the chunk
    expressions before printing.

    With fused=True, 1st and pure-2nd derivative leaves move from pp_arrays
    to the stencil buckets (mixed 2nd derivatives stay precomputed --
    measured 1.9x slower when fused; see project_mixed_deriv_fusion).
    """
    chunk_outs = set()
    cse_temps = set()
    for c in result.chunks:
        chunk_outs |= set(c.outputs.keys())
        for s, _ in c.cse_temps:
            cse_temps.add(str(s))

    syms = set()
    for c in result.chunks:
        for _, e in c.outputs.items():
            if hasattr(e, "free_symbols"):
                syms |= e.free_symbols
        for _, e in c.cse_temps:
            if hasattr(e, "free_symbols"):
                syms |= e.free_symbols

    pp_arrays, idx_consts, scalars = [], [], []
    stencil_1st, stencil_2nd = [], []
    # `scalar_syms` keeps the original Symbol objects (with their assumptions)
    # so callers can rename them via xreplace without an assumptions mismatch.
    scalar_syms = []
    rename_map = {}
    grad1_re = re.compile(r"^a?grad_([0-2])_(\w+)$")
    grad2_re = re.compile(r"^grad2_([0-2])_([0-2])_(\w+)$")
    for s in sorted(syms, key=lambda x: x.name):
        n = s.name
        if n in chunk_outs or n in cse_temps:
            continue
        if n.endswith("[pp]"):
            arr = n[:-4]
            if fused:
                m1 = grad1_re.match(arr)
                m2 = grad2_re.match(arr)
                if m1:
                    stencil_1st.append((arr, int(m1.group(1)), m1.group(2)))
                    rename_map[s] = sym.Symbol(f"v_{arr}")
                    continue
                if m2 and m2.group(1) == m2.group(2):
                    stencil_2nd.append((arr, int(m2.group(1)), m2.group(3)))
                    rename_map[s] = sym.Symbol(f"v_{arr}")
                    continue
                # mixed 2nd derivs (and everything else) stay precomputed
            pp_arrays.append(arr)
            rename_map[s] = sym.Symbol(f"v_{arr}")
        else:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]$", n)
            if m:
                base, idx = m.group(1), int(m.group(2))
                idx_consts.append((base, idx))
                rename_map[s] = sym.Symbol(f"{base}_{idx}")
            else:
                scalars.append(n)
                scalar_syms.append(s)
                # rename will be set by callers that broadcast-as-VEC
    return {
        "pp_arrays": pp_arrays,
        "idx_consts": idx_consts,
        "scalars": scalars,
        "scalar_syms": scalar_syms,
        "stencil_1st": stencil_1st,
        "stencil_2nd": stencil_2nd,
        "rename_map": rename_map,
    }


def _lazy_reorder(lines):
    """Move each prologue `const VEC` declaration to just before the first
    chunk that references it (first-use placement -- information only the
    chunk structure provides). Measured +2.6% on fused AVX-512 BSSN."""
    chunk_idx = [i for i, l in enumerate(lines) if l.startswith("// === L:")]
    if not chunk_idx:
        return lines
    first_chunk = chunk_idx[0]
    decl_re = re.compile(r"^const VEC (\w+) = ")
    movable, head = {}, []
    for l in lines[:first_chunk]:
        m = decl_re.match(l)
        (movable.__setitem__(m.group(1), l) if m else head.append(l))
    body = lines[first_chunk:]
    ci, chunk_of = -1, []
    for l in body:
        if l.startswith("// === L:"):
            ci += 1
        chunk_of.append(ci)
    name_re = re.compile(r"\b(v_\w+|lambda_\d|lambda_f_\d)\b")
    first_use = {}
    for l, ci in zip(body, chunk_of):
        for n in name_re.findall(l):
            if n in movable and n not in first_use:
                first_use[n] = ci
    per_chunk = {}
    for n, l in movable.items():
        per_chunk.setdefault(first_use.get(n, 0), []).append(l)
    out, ci = head[:], -1
    for l in body:
        if l.startswith("// === L:"):
            ci += 1
            out.append(l)
            out.extend(per_chunk.get(ci, []))
            continue
        out.append(l)
    return out


def _emit_avx_prologue(leaves):
    """Emit prologue lines: VLOAD for pp arrays, VSET broadcast for idx
    consts, and (fused mode) inline 6th-order stencils for 1st / pure-2nd
    derivative leaves. The chunk body is identical either way -- fusion is
    purely a prologue swap."""
    from dendrosym.cascade.common import simd_stencil_1st, simd_stencil_2nd_pure

    lines = ["// --- IR-AVX prologue: load leaves into VEC ---"]
    if leaves.get("stencil_1st") or leaves.get("stencil_2nd"):
        lines += [
            "// Fused mode: 1st/pure-2nd derivs via inline stencils (mixed",
            "// 2nd derivs remain precomputed). Wrapper provides hx/hy/hz,",
            "// nx, ny; reads reach pp +- 3*stride (valid for pw >= 3).",
            "const double idx60    = (1.0/60.0)  / hx;",
            "const double idy60    = (1.0/60.0)  / hy;",
            "const double idz60    = (1.0/60.0)  / hz;",
            "const double idx2_180 = (1.0/180.0) / (hx*hx);",
            "const double idy2_180 = (1.0/180.0) / (hy*hy);",
            "const double idz2_180 = (1.0/180.0) / (hz*hz);",
        ]
    for arr in leaves["pp_arrays"]:
        lines.append(f"const VEC v_{arr} = VLOAD({arr}+pp);")
    for arr, d, var in leaves.get("stencil_1st", []):
        lines.append(f"const VEC v_{arr} = {simd_stencil_1st(d, var)};")
    for arr, d, var in leaves.get("stencil_2nd", []):
        lines.append(f"const VEC v_{arr} = {simd_stencil_2nd_pure(d, var)};")
    for base, idx in leaves["idx_consts"]:
        # lambda[i] is unsigned int; lambda_f[i] is double; cast to double.
        lines.append(f"const VEC {base}_{idx} = VSET((double)({base}[{idx}]));")
    # eta is already a VEC in scope (set by the wrapper).
    return lines


def _emit_avx_chunks(result, leaves, fma=False, split=1):
    """Emit each chunk as VEC declarations using VecPrinter, with leaves renamed.

    fma=True emits tree-level VFMA/VFNMADD for every product term; split=k>1
    breaks long sums into k independent FMA chains joined by a balanced tree."""
    from dendrosym.cascade.vec_printer import to_vec_cpp

    rename = leaves["rename_map"]

    rhs_names = set()
    for c in result.chunks:
        for nm in c.outputs:
            if "_rhs" in nm:
                rhs_names.add(nm)

    lines = []
    for c in result.chunks:
        lines.append("")
        lines.append(f"// === L: {c.name} ({c.n_temps} temps, {len(c.outputs)} outputs) ===")
        # CSE temps first, in order
        for s, e in c.cse_temps:
            e2 = e.xreplace(rename) if rename else e
            lines.append(f"const VEC {s.name} = {to_vec_cpp(e2, fma=fma, split=split)};")
        for name, e in c.outputs.items():
            e2 = e.xreplace(rename) if rename else e
            if name in rhs_names:
                lines.append(f"VSTORE({name}+pp, {to_vec_cpp(e2, fma=fma, split=split)});")
            else:
                lines.append(f"const VEC {name} = {to_vec_cpp(e2, fma=fma, split=split)};")
    return lines


def _emit_avx_global_cse(result, leaves, fma=False, split=1):
    """Emit the global symbol-aware CSE node list (one CSE across all chunks,
    named tensors atomic) as flat VEC declarations. Removes the cross-chunk
    recompute that per-chunk CSE leaves behind -- fewer ops, which can help the
    throughput/instruction-bound SIMD regimes (AVX2, large patches)."""
    from dendrosym.cascade.vec_printer import to_vec_cpp

    rename = leaves["rename_map"]
    nodes = result.global_cse_result()
    lines = ["", "// === global symbol-aware CSE (cross-chunk sharing) ==="]
    for name, e, kind in nodes:
        e2 = e.xreplace(rename) if rename else e
        if kind == "rhs":
            lines.append(f"VSTORE({name}+pp, {to_vec_cpp(e2, fma=fma, split=split)});")
        else:
            lines.append(f"const VEC {name} = {to_vec_cpp(e2, fma=fma, split=split)};")
    return lines


_SIMD_HEADERS = {
    "avx2": (
        "__m256d", 4,
        "_mm256_loadu_pd", "_mm256_storeu_pd", "_mm256_set1_pd",
        "_mm256_add_pd", "_mm256_sub_pd", "_mm256_mul_pd", "_mm256_div_pd",
        "_mm256_fmadd_pd", "_mm256_sqrt_pd",
    ),
    "avx512": (
        "__m512d", 8,
        "_mm512_loadu_pd", "_mm512_storeu_pd", "_mm512_set1_pd",
        "_mm512_add_pd", "_mm512_sub_pd", "_mm512_mul_pd", "_mm512_div_pd",
        "_mm512_fmadd_pd", "_mm512_sqrt_pd",
    ),
    "scalar": (
        "double", 1,
        "(*)", "(*=)", "(double)",
        "+", "-", "*", "/",
        "(a*b+c)", "std::sqrt",
    ),
}


def _macro_defines(simd: str) -> list:
    """Inline macro defines; same shape as cascade_common headers but inlined
    into a generated kernel so we don't depend on external macro state.

    Also provides VLOG / VEXP per-lane fallbacks: AVX2/AVX-512 don't have
    standard log/exp intrinsics (only via Intel SVML), so we extract lanes,
    call libm scalar log/exp, and repack. This is slow but PDE kernels
    rarely use these on the hot path.
    """
    typ, _w, vload, vstore, vset, vadd, vsub, vmul, vdiv, vfma, vsqrt = _SIMD_HEADERS[simd]
    if simd == "scalar":
        return [
            "#include <cmath>",
            "#define VEC double",
            "#define VLOAD(p)     (*(p))",
            "#define VSTORE(p, v) (*(p) = (v))",
            "#define VSET(x)      ((double)(x))",
            "#define VADD(a, b)   ((a) + (b))",
            "#define VSUB(a, b)   ((a) - (b))",
            "#define VMUL(a, b)   ((a) * (b))",
            "#define VDIV(a, b)   ((a) / (b))",
            "#define VFMA(a, b, c) ((a) * (b) + (c))",
            "#define VSQRT(a)     (std::sqrt(a))",
            "#define VLOG(a)      (std::log(a))",
            "#define VEXP(a)      (std::exp(a))",
        ]
    # SIMD: per-lane log/exp helpers. Inline static, defined in TU.
    lanes = 4 if simd == "avx2" else 8
    align = 32 if simd == "avx2" else 64
    store = "_mm256_store_pd" if simd == "avx2" else "_mm512_store_pd"
    load = "_mm256_load_pd" if simd == "avx2" else "_mm512_load_pd"
    helpers = [
        "#include <immintrin.h>",
        "#include <cmath>",
        f"static inline {typ} _vlog_lanes({typ} x) {{",
        f"    alignas({align}) double _a[{lanes}]; {store}(_a, x);",
        f"    for (int _i = 0; _i < {lanes}; _i++) _a[_i] = std::log(_a[_i]);",
        f"    return {load}(_a);",
        "}",
        f"static inline {typ} _vexp_lanes({typ} x) {{",
        f"    alignas({align}) double _a[{lanes}]; {store}(_a, x);",
        f"    for (int _i = 0; _i < {lanes}; _i++) _a[_i] = std::exp(_a[_i]);",
        f"    return {load}(_a);",
        "}",
    ]
    return helpers + [
        f"#define VEC {typ}",
        f"#define VLOAD(p)     {vload}(p)",
        f"#define VSTORE(p, v) {vstore}((p), (v))",
        f"#define VSET(x)      {vset}(x)",
        f"#define VADD(a, b)   {vadd}((a), (b))",
        f"#define VSUB(a, b)   {vsub}((a), (b))",
        f"#define VMUL(a, b)   {vmul}((a), (b))",
        f"#define VDIV(a, b)   {vdiv}((a), (b))",
        f"#define VFMA(a, b, c) {vfma}((a), (b), (c))",
        f"#define VSQRT(a)     {vsqrt}(a)",
        "#define VLOG(a)      _vlog_lanes(a)",
        "#define VEXP(a)      _vexp_lanes(a)",
    ]


def emit_jax_kernel(result, jit: bool = True):
    """Build a JAX-compatible forward kernel from a CascadeResult.

    Parameters
    ----------
    result : CascadeResult
        Output of CascadeBuilder.build() (or any post-build transform).
    jit : bool
        Wrap the returned callable in `jax.jit` so the first invocation
        triggers XLA compilation. Default True.

    Returns
    -------
    callable kernel(inputs: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]
        Each input key is a leaf symbol name (e.g. 'alpha[pp]', 'a[pp]', 'mu').
        Each output key is one of the chunk's named outputs (e.g. 'a_rhs',
        'P00_out'). Non-output CSE temps are not exposed.

    Forward only; differentiation through the cascade hasn't been verified
    (each chunk's lambdified callable is pure, so `jax.grad` should work in
    principle but isn't tested).

    Note: each chunk's outputs become inputs to subsequent chunks via Symbol
    references, so the free-symbol set of a chunk's expressions includes
    both leaves and prior-chunk-output names. The kernel walks chunks in
    order, populating a working state dict.
    """
    import sympy as sym
    try:
        import jax
        import jax.numpy as jnp  # noqa: F401  (registers numpy backends)
    except ImportError as e:
        raise RuntimeError(
            "JAX is required for emit_jax_kernel. Install with `pip install jax`."
        ) from e

    # Pre-lambdify every CSE temp and every output, recording the free-symbol
    # name list so we can pull from the state dict at runtime.
    chunk_lambdas = []
    for c in result.chunks:
        temps = []
        for s, e in c.cse_temps:
            free_syms = list(e.free_symbols)
            free_names = [fs.name for fs in free_syms]
            f = sym.lambdify(free_syms, e, modules='jax')
            temps.append((s.name, free_names, f))
        outs = []
        for name, e in c.outputs.items():
            free_syms = list(e.free_symbols)
            free_names = [fs.name for fs in free_syms]
            f = sym.lambdify(free_syms, e, modules='jax')
            outs.append((name, free_names, f))
        chunk_lambdas.append((c.name, temps, outs))

    output_names = [name for c in result.chunks for name in c.outputs]

    def kernel_impl(inputs):
        # Working scope: leaves + already-computed chunk outputs.
        state = dict(inputs)
        for chunk_name, temps, outs in chunk_lambdas:
            # CSE temps live within the chunk; emit into a local scope so they
            # don't pollute downstream chunks. Outputs go into `state` (visible
            # to later chunks) AND into `local` (so later outputs in the same
            # chunk that reference earlier outputs by name resolve correctly).
            # The latter matters for smart-split's `<chunk>_shared` sub-chunks,
            # whose 100+ outputs are post-CSE temps that reference each other
            # in CSE topological order.
            local = dict(state)
            for tname, free_names, f in temps:
                local[tname] = f(*[local[n] for n in free_names])
            for oname, free_names, f in outs:
                value = f(*[local[n] for n in free_names])
                state[oname] = value
                local[oname] = value
        return {n: state[n] for n in output_names}

    if jit:
        return jax.jit(kernel_impl)
    return kernel_impl


def emit_kernel_function_cpp(
    result, fn_name: str, simd: str = "avx2",
    fold_vfma_passes: int = 0,
) -> str:
    """Auto-generate a complete C++ kernel function from a CascadeResult.

    Convention: leaf symbols whose name ends in `[pp]` become per-point input
    arrays (the function takes a `const double*` pointer and the kernel does
    VLOAD at offset `pp` per batch). All other leaves are scalar arguments
    passed as `double` and VSET-broadcast inside the kernel.

    Output names containing `_out` are flagged as output arrays (`double*`
    arguments, VSTORE'd). All other outputs are intermediate `const VEC`
    declarations.

    The emitted function takes (int N, ...inputs..., ...outputs...) and runs
    a 4-wide (avx2) or 8-wide (avx512) batched loop with shift-back tail.

    No PDE-specific harness coupling. Hand the result to a small caller and
    you're done.
    """
    from dendrosym.cascade.vec_printer import to_vec_cpp
    if simd not in ("avx2", "avx512", "scalar"):
        raise ValueError(simd)
    _, lane_width, *_ = _SIMD_HEADERS[simd]
    leaves = _classify_leaves(result)
    rename = leaves["rename_map"]
    pp_arrays = leaves["pp_arrays"]
    idx_consts = leaves["idx_consts"]
    scalars = leaves["scalars"]

    # Output names → split into `_out`-suffixed arrays vs intermediates.
    out_arrays = []
    rhs_names = set()
    for c in result.chunks:
        for nm in c.outputs:
            if nm.endswith("_out") or "_rhs" in nm:
                rhs_names.add(nm)
                if nm not in out_arrays:
                    out_arrays.append(nm)

    # Function signature
    arg_lines = ["    int N,"]
    for arr in pp_arrays:
        arg_lines.append(f"    const double *__restrict__ {arr},")
    for base, idx in idx_consts:
        # idx-style consts are flat-index-into-array: caller passes the array.
        # We just take a `double` parameter named `<base>_<idx>` to keep it
        # simple and broadcast it per-batch.
        arg_lines.append(f"    double {base}_{idx},")
    for name in scalars:
        arg_lines.append(f"    double {name},")
    for arr in out_arrays:
        arg_lines.append(f"    double *__restrict__ {arr},")
    if arg_lines[-1].endswith(","):
        arg_lines[-1] = arg_lines[-1][:-1]

    # Body emission
    body_lines = []
    body_lines.append("// --- per-batch leaf loads ---")
    for arr in pp_arrays:
        body_lines.append(f"const VEC v_{arr} = VLOAD({arr}+pp);")
    for base, idx in idx_consts:
        body_lines.append(f"const VEC {base}_{idx}_v = VSET({base}_{idx});")
    for name in scalars:
        body_lines.append(f"const VEC {name}_v = VSET({name});")
    # Build a rename map: scalars/idx_consts get a `_v` suffix when used as VEC.
    rename_for_kernel = dict(rename)
    for base, idx in idx_consts:
        rename_for_kernel[sym.Symbol(f"{base}[{idx}]")] = sym.Symbol(f"{base}_{idx}_v")
    # scalars: use the original Symbol objects (preserving assumptions) so
    # xreplace matches the symbols actually present in the expressions.
    for orig_sym in leaves["scalar_syms"]:
        rename_for_kernel[orig_sym] = sym.Symbol(f"{orig_sym.name}_v")

    for c in result.chunks:
        body_lines.append("")
        body_lines.append(f"// === {c.name} ({c.n_temps} temps, {len(c.outputs)} outputs) ===")
        for s, e in c.cse_temps:
            e2 = e.xreplace(rename_for_kernel) if rename_for_kernel else e
            body_lines.append(f"const VEC {s.name} = {to_vec_cpp(e2)};")
        for nm, e in c.outputs.items():
            e2 = e.xreplace(rename_for_kernel) if rename_for_kernel else e
            if nm in rhs_names:
                body_lines.append(f"VSTORE({nm}+pp, {to_vec_cpp(e2)});")
            else:
                body_lines.append(f"const VEC {nm} = {to_vec_cpp(e2)};")

    # Assemble
    L = len(result.chunks)
    if L == 7:
        kind = "natural"
    elif L > 7:
        kind = "split from natural"
    else:
        kind = "collapsed from natural"

    out = []
    out.append(f"// Auto-generated kernel: {fn_name}")
    out.append(f"// L = {L} chunks ({kind}); SIMD = {simd} ({lane_width}-wide).")
    out.append(f"// VFMA folding passes: {fold_vfma_passes}")
    out.append("")
    out += _macro_defines(simd)
    out.append("")
    out.append(f"static inline void {fn_name}(")
    out.extend(arg_lines)
    out.append(") {")
    out.append(f"    int i = 0;")
    out.append(f"    for (; i + {lane_width} <= N; i += {lane_width}) {{")
    out.append(f"        const int pp = i;")
    out.append("        {")
    out += ["            " + ln for ln in body_lines]
    out.append("        }")
    out.append("    }")
    out.append("    // shift-back tail batch (kernel deterministic per-pp)")
    out.append(f"    if (i < N && N >= {lane_width}) {{")
    out.append(f"        const int pp = N - {lane_width};")
    out.append("        {")
    out += ["            " + ln for ln in body_lines]
    out.append("        }")
    out.append("    }")
    out.append("}")
    out.append("")
    # Avoid leaking macros into the rest of the TU.
    out.append("#undef VEC")
    out.append("#undef VLOAD")
    out.append("#undef VSTORE")
    out.append("#undef VSET")
    out.append("#undef VADD")
    out.append("#undef VSUB")
    out.append("#undef VMUL")
    out.append("#undef VDIV")
    out.append("#undef VFMA")
    out.append("#undef VSQRT")
    text = "\n".join(out) + "\n"
    if fold_vfma_passes > 0 and simd != "scalar":
        from dendrosym.cascade.vec_printer import fold_vfma
        text = fold_vfma(text, max_passes=fold_vfma_passes)
    return text



# ---------------------------------------------------------------------------
# Generic, option-driven composition of the pieces above (system-agnostic).
#
# emit_body() is the exact statement sequence the BSSN front-end
# (systems/bssn/emit.py) performed after build_ir(); it is parameterised by
# CascadeOptions so ANY system reaches the paper's knobs (inline threshold,
# fused stencils, tree FMA, global CSE, lazy prologue) from one call. The BSSN
# front-end is re-expressed on top of it and the regen oracle
# (scripts/regen_vikr_kernels.sh) pins the bytes.
# ---------------------------------------------------------------------------
_VEC_MACROS = ("VEC", "VLOAD", "VSTORE", "VSET", "VADD", "VSUB", "VMUL", "VDIV",
               "VFMA", "VFNMADD", "VSQRT", "VLOG", "VEXP", "VMASK", "VLOADM", "VSTOREM", "VPOW")
_VFNMADD = {
    "scalar": "#define VFNMADD(a, b, c) ((c) - (a) * (b))",
    "avx2": "#define VFNMADD(a, b, c) _mm256_fnmadd_pd((a), (b), (c))",
    "avx512": "#define VFNMADD(a, b, c) _mm512_fnmadd_pd((a), (b), (c))",
}


def emit_body(result, options, header: str = "", vec=None) -> str:
    """CascadeResult -> C++ body text under CascadeOptions.

    SIMD (options.simd in avx2/avx512): the VEC-macro body meant to be
    ``#include``d inside a wrapper loop that provides ``pp``, the VEC macros,
    and every bare-scalar leaf as a VEC in scope. Same text for avx2 and
    avx512 (only the wrapper's macro set differs).

    Scalar: the unrolled ``const double`` body (Dendro naming, ``[pp]`` arrays).

    ``header`` is placed verbatim before the body (end it with a newline).
    ``vec`` forces the VEC-macro form even for options.simd == "scalar" (the
    body then compiles per point under the width-1 macro set, VEC=double).

    Store contract (vikr's): outputs whose name contains ``_rhs`` are
    VSTORE'd / assigned to ``name[pp]``; every other output is a named local.
    """
    from dendrosym.cascade.vec_printer import fold_vfma

    if options.inline_threshold > 0:
        result = _inline_low_use_temps(result, threshold=options.inline_threshold)
    use_vec = (options.simd != "scalar") if vec is None else vec
    if not use_vec:
        if options.fused:
            raise NotImplementedError("fused=True requires the VEC emitter")
        if options.global_cse:
            body = result.emit_cpp_global_cse(dendro_var_style=True, short_names=True)
        else:
            body = result.emit_cpp_unrolled(
                dendro_var_style=True, inline_threshold=0, short_names=False)
        return header + body + "\n"

    leaves = _classify_leaves(result, fused=options.fused)
    lines = header.splitlines() if header else []
    lines += _emit_avx_prologue(leaves)
    if options.global_cse:
        lines += _emit_avx_global_cse(result, leaves, fma=options.fma_tree,
                                      split=options.fma_split)
    else:
        lines += _emit_avx_chunks(result, leaves, fma=options.fma_tree,
                                  split=options.fma_split)
    if options.lazy_prologue:
        lines = _lazy_reorder(lines)
    text = "\n".join(lines) + "\n"
    if options.vfma > 0:
        text = fold_vfma(text, max_passes=options.vfma)
    return text


# masked-lane macros for a wrapper's partial (tail) batch: VMASK(n) builds the
# mask for the first n lanes; VLOADM/VSTOREM never touch masked-off lanes, so a
# batch may extend past the valid points without reading garbage (uninitialised
# workspace padding is often denormal -> microcode assists) or writing outside.
# per-lane pow for exponents the printer cannot reduce to VSQRT/VMUL (adapter side,
# same shape as the engine's _vlog_lanes/_vexp_lanes helpers).
def _vpow_lines(simd):
    if simd == "scalar":
        return ["#define VPOW(a, b)    (std::pow((a), (b)))"]
    typ, lanes, align = (("__m256d", 4, 32) if simd == "avx2" else ("__m512d", 8, 64))
    store = "_mm256_store_pd" if simd == "avx2" else "_mm512_store_pd"
    load = "_mm256_load_pd" if simd == "avx2" else "_mm512_load_pd"
    return [f"static inline {typ} _vpow_lanes({typ} x, {typ} y) {{",
            f"    alignas({align}) double _a[{lanes}], _b[{lanes}]; {store}(_a, x); {store}(_b, y);",
            f"    for (int _i = 0; _i < {lanes}; _i++) _a[_i] = std::pow(_a[_i], _b[_i]);",
            f"    return {load}(_a);", "}",
            "#define VPOW(a, b)    _vpow_lanes((a), (b))"]


_MASKED = {
    "scalar": ["#define VMASK(n)      (n)",
               "#define VLOADM(p)     (*(p))",
               "#define VSTOREM(p, v) (*(p) = (v))"],
    "avx2": ["#define VMASK(n)      _mm256_set_epi64x((n) > 3 ? -1LL : 0LL, (n) > 2 ? -1LL : 0LL, "
             "(n) > 1 ? -1LL : 0LL, (n) > 0 ? -1LL : 0LL)",
             "#define VLOADM(p)     _mm256_maskload_pd((p), __cascade_mask)",
             "#define VSTOREM(p, v) _mm256_maskstore_pd((p), __cascade_mask, (v))"],
    "avx512": ["#define VMASK(n)      ((__mmask8)((1u << (n)) - 1u))",
               "#define VLOADM(p)     _mm512_maskz_loadu_pd(__cascade_mask, (p))",
               "#define VSTOREM(p, v) _mm512_mask_storeu_pd((p), __cascade_mask, (v))"],
}


def macro_block(simd: str) -> list:
    """File-scope macro set for a wrapper: _macro_defines(simd) + VFNMADD
    (tree FMA emits VFNMADD; the vikr wrappers define it by hand) + the
    masked-lane macros VMASK/VLOADM/VSTOREM for partial batches."""
    return _macro_defines(simd) + [_VFNMADD[simd]] + _MASKED[simd] + _vpow_lines(simd)


def masked_body(body: str) -> str:
    """The same VEC body with every VLOAD/VSTORE turned into its masked form
    (a text transform on the emitted body; the engine is untouched)."""
    return re.sub(r"\bVSTORE\(", "VSTOREM(", re.sub(r"\bVLOAD\(", "VLOADM(", body))


def scalar_macro_block() -> list:
    """Width-1 macro set usable INSIDE a function (no #include lines): the
    same VEC body compiles per point with VEC=double. Preceded by undefs."""
    return undef_block() + [l for l in _macro_defines("scalar")
                            if not l.startswith("#include")] + [_VFNMADD["scalar"]] + _MASKED["scalar"] + _vpow_lines("scalar")


def undef_block() -> list:
    return [f"#undef {m}" for m in _VEC_MACROS]


def kernel_signature(result, options, fused_ok=False):
    """Classify leaves/outputs of `result` into the argument groups a
    standalone kernel (or a host wrapper) must provide. Returns a dict:
    pp_arrays, idx_consts [(base, idx)], scalars, outputs (names containing
    ``_rhs`` or ending ``_out``), idx_bases (unique bases)."""
    leaves = _classify_leaves(result, fused=options.fused and fused_ok)
    outputs = []
    for c in result.chunks:
        for nm in c.outputs:
            if (nm.endswith("_out") or "_rhs" in nm) and nm not in outputs:
                outputs.append(nm)
    bases = []
    for b, _i in leaves["idx_consts"]:
        if b not in bases:
            bases.append(b)
    return {"pp_arrays": leaves["pp_arrays"], "idx_consts": leaves["idx_consts"],
            "idx_bases": bases, "scalars": leaves["scalars"], "outputs": outputs,
            "stencil_1st": leaves["stencil_1st"], "stencil_2nd": leaves["stencil_2nd"]}


def emit_standalone_kernel(result, options, header: str = "") -> str:
    """Complete C++ translation unit around emit_body(): macro set, a
    ``static inline void <fn_name>(int N, ...)`` taking one ``const double*``
    per ``X[pp]`` leaf, one ``const double*`` per indexed-constant base
    (``lambda``), one ``double`` per bare scalar (``<name>_s``), one
    ``double*`` per output; W-wide loop with shift-back tail and a width-1
    fallback for N < W (scalar: a plain per-point loop, VEC=double).

    Outputs: names containing ``_rhs`` are stored by the body itself;
    names ending in ``_out`` are named locals the wrapper stores after the
    body (their pointer parameter is ``<name>_ptr``). Any other object is an
    intermediate.

    Unlike emit_kernel_function_cpp (kept for its callers), this honours every
    CascadeOptions knob because the body IS emit_body().
    """
    if options.fused:
        raise NotImplementedError("fused stencils need a grid-aware wrapper "
                                  "(nx/ny/hx.. in scope); use emit_body() in your own loop")
    sig = kernel_signature(result, options)
    body = emit_body(result, options, header=header, vec=True)
    W = options.width
    rhs_outs = [o for o in sig["outputs"] if "_rhs" in o]
    out_outs = [o for o in sig["outputs"] if "_rhs" not in o]
    args = ["    int N"]
    args += [f"    const double *__restrict__ {a}" for a in sig["pp_arrays"]]
    args += [f"    const double *__restrict__ {b}" for b in sig["idx_bases"]]
    args += [f"    double {s}_s" for s in sig["scalars"]]
    args += [f"    double *__restrict__ {o}" for o in rhs_outs]
    args += [f"    double *__restrict__ {o}_ptr" for o in out_outs]
    prologue = [f"const VEC {s} = VSET({s}_s);" for s in sig["scalars"]]
    epilogue = [f"VSTORE({o}_ptr+pp, {o});" for o in out_outs]

    def block(indent):
        out = [indent + l for l in prologue]
        out.append(indent + "{")
        out += [indent + "    " + l if l else "" for l in body.rstrip("\n").split("\n")]
        out += [indent + "    " + l for l in epilogue]
        out.append(indent + "}")
        return out

    out = [f"// Auto-generated cascade kernel: {options.fn_name}",
           f"// L = {len(result.chunks)} chunks; SIMD = {options.simd} ({W}-wide); "
           f"inline_threshold={options.inline_threshold} fma_tree={options.fma_tree} "
           f"global_cse={options.global_cse}", ""]
    out += macro_block(options.simd)
    out += ["", f"static inline void {options.fn_name}(", ",\n".join(args) + ") {"]
    if W > 1:
        out += ["    int i = 0;",
                f"    if (N >= {W}) {{",
                f"        for (; i + {W} <= N; i += {W}) {{",
                "            const int pp = i;"]
        out += block("            ")
        out += ["        }",
                "        // shift-back tail: recompute the overlap (deterministic per pp)",
                "        if (i < N) {",
                f"            const int pp = N - {W};"]
        out += block("            ")
        out += ["        }", "    } else {",
                "        // fewer points than lanes: same body, width-1 macro set"]
        out += ["        " + l for l in scalar_macro_block()]
        out += ["        for (int pp = 0; pp < N; pp++) {"]
        out += block("            ")
        out += ["        }", "    }"]
    else:
        out += ["    for (int pp = 0; pp < N; pp++) {"]
        out += block("        ")
        out += ["    }"]
    out += ["}", ""] + undef_block()
    return "\n".join(out) + "\n"
