"""dendro_bridge -- emit a polynomial-cascade RHS kernel from a DendroConfiguration.

This is the SymPyGR side of the fold-in. The cascade ENGINE (builder/emit) is
untouched and system-agnostic; this module

  1. builds the cascade IR from a config's own RHS expressions and a
     user-registered layer spec (``config.set_cascade_spec_function``),
  2. emits the VEC-macro body exactly as vikr's emitters would for that IR
     (``emit_body``), and
  3. writes the small C++ *adapter* files the Dendro-6 ``rhs.cpp`` wrapper
     needs: an alias table mapping vikr's bare leaf names (``alpha``,
     ``grad_0_alpha``, ``alpha_rhs``) onto Dendro-6 storage (``in.alpha``,
     ``d.alpha_x``, ``out.alpha``), a per-batch prologue (bare scalar params
     broadcast to VEC), the macro sets, and a JSON manifest.

Nothing here rewrites the body: the body is what the engine emits. The flat
CSE path is not touched either (flag off => byte-identical gencode).

Contract with the config (Route C of vikr's cascade_api_guide.md):
  spec_func(rhs_pairs) -> (chunks, leaves) where rhs_pairs is the flat
  ``[(name_rhs, expr), ...]`` list and chunks is ``[(layer_name, OrderedDict)]``
  whose LAST layer is the RHS assembly (its keys == the rhs names). Earlier
  layers are the config's own named tensors (igt, C1..C3, Ricci, ...) taken
  straight from ``dendrosym.nr``; the bridge aligns their derivative form with
  the post-``find_derivatives`` RHS so by-value substitution finds them.
"""
from __future__ import annotations

import dataclasses
import json
import re
from collections import OrderedDict
from pathlib import Path

import sympy as sym
from sympy.core.function import AppliedUndef, UndefinedFunction

CASCADE_BRIDGE_VERSION = "b4"   # b4: compile-time width selection (select file);   # b2: masked tail batches (body_tail); b3: hybrid tail + half-integer pow printer          # part of the gencode cache discriminator
CASCADE_TEMPLATE_SUPPORT = {"evolution", "constraint"}   # var_types whose template has the wrapper branch
DERIV_FUNCS = ("grad", "grad2", "agrad", "kograd")
COORD_LEAVES = ("x", "y", "z", "r_coord", "t")

# identifiers the generated rhs.cpp wrapper already uses in scope (rhs.cpp.j2),
# plus the struct names and our own prefix. Cascade names must not collide.
RESERVED_WRAPPER_NAMES = frozenset("""
in out d pp i j k x y z r_coord t nx ny nz hx hy hz PW n idx bflag offset pmin pmax sz
BLK_SZ bytes deriv_base unzipVarsRHS uZipVars uZipConstVars sigma SOLVER_DERIVS
__ko_row __ko_rows __r __bc_row __bc_rows __mem_pool VEC
""".split()) | frozenset("""
alignas alignof and and_eq asm auto bitand bitor bool break case catch char char16_t
char32_t class compl const constexpr const_cast continue decltype default delete do
double dynamic_cast else enum explicit export extern false float for friend goto if
inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq
private protected public register reinterpret_cast return short signed sizeof static
static_assert static_cast struct switch template this thread_local throw true try
typedef typeid typename union unsigned using virtual void volatile wchar_t while xor
xor_eq
""".split())


class CascadeNamingError(ValueError):
    """A cascade name cannot be mapped onto (or collides with) the wrapper."""


@dataclasses.dataclass
class CascadeParts:
    body: str            # width-agnostic VEC body (what the engine emits)
    body_tail: str       # same body with masked loads/stores (partial batches)
    alias: str
    prologue: str
    select: str          # compile-time ISA selection -> includes one macro set, defines __cascade_W
    macros_avx2: str
    macros_avx512: str
    macros_scalar: str
    macros_undef: str
    manifest: dict


# ---------------------------------------------------------------------------
# 1. IR from the config
# ---------------------------------------------------------------------------
def derivs_to_symbols(expr, idx_str: str = "[pp]"):
    """``grad(i, X[pp])`` -> ``Symbol('grad_i_X[pp]')`` etc.

    SymPyGR keeps derivatives as applications of the undefined functions
    ``nr.d``/``nr.d2s``/``nr.ad`` (``grad``, ``grad2``, ``agrad``); vikr's IR
    expects the leaf SYMBOLS its emitters classify by name. ``grad2`` indices
    are canonicalised ascending, matching ``codegen.change_deriv_names`` and
    ``find_all_unique_ders`` so the alias targets exist in the ``d.`` struct.
    """
    reps = {}
    for f in expr.atoms(AppliedUndef):
        name = f.func.__name__
        if name not in DERIV_FUNCS:
            continue
        *idx, var = f.args
        if not isinstance(var, sym.Symbol):
            raise CascadeNamingError(
                f"derivative of a non-symbol survived expansion: {f} -- the cascade "
                "needs replace_and_expand_derivatives=True (no staged compound derivs)")
        if not var.name.endswith(idx_str):
            raise CascadeNamingError(f"derivative of a non-field symbol: {f}")
        idx = [int(i) for i in idx]
        if name == "grad2":
            idx = sorted(idx)
        base = var.name[: -len(idx_str)]
        reps[f] = sym.Symbol(f"{name}_{'_'.join(str(i) for i in idx)}_{base}{idx_str}")
    return expr.xreplace(reps) if reps else expr


def _needs_restore():
    import dendrosym.nr as nr
    return (type(nr.d) is not UndefinedFunction
            or type(nr.d2s) is not UndefinedFunction)


def build_config_cascade(config, var_type, spec_func, options):
    """Config + registered spec -> CascadeResult (the IR).

    Steps: take the post-``find_derivatives`` RHS list (same object the flat
    path CSEs), hand ``[(name, expr)]`` to the spec, align every non-assembly
    layer to the RHS's derivative form (restore symbols + expand compound
    derivatives, exactly the transforms ``find_derivatives`` applied to the
    RHS), convert derivative applications to leaf Symbols, then build.
    """
    from dendrosym.cascade.builder import build_cascade_ir
    from dendrosym.derivs import restore_only_symbols
    from dendrosym.general_configs import DendroConfiguration

    if var_type not in CASCADE_TEMPLATE_SUPPORT:
        raise NotImplementedError(
            f"cascade kernels are wired for var_types {sorted(CASCADE_TEMPLATE_SUPPORT)}; "
            f"{var_type!r} has no wrapper branch in its template yet")
    if config.idx_str != "[pp]":
        raise CascadeNamingError("the cascade emitters assume idx_str == '[pp]'")
    if not getattr(config, "replace_and_expand_derivatives", True):
        raise NotImplementedError("cascade + staged compound derivatives is not supported")
    if options.fused:
        raise NotImplementedError(
            "fused stencils in the Dendro-6 wrapper are a follow-on (needs interior-only "
            "handling and dropping the 1st/pure-2nd d.* buffers)")

    if config.stored_rhs_function.get(var_type) is None:
        config.find_derivatives(var_type)
    stored = config.stored_rhs_function[var_type]
    all_exp = list(stored["exprs"])
    names = list(stored["all_rhs_names"])
    restore = _needs_restore()
    if restore:
        all_exp = [restore_only_symbols(e) for e in all_exp]

    chunks, _leaves_ignored = spec_func(list(zip(names, all_exp)))
    chunks = [(nm, OrderedDict(od)) for nm, od in chunks]
    if not chunks or list(chunks[-1][1].keys()) != names:
        raise CascadeNamingError(
            "the spec's LAST layer must be the RHS assembly with keys == the rhs names "
            f"(in order); got {list(chunks[-1][1].keys())[:5]}... vs {names[:5]}...")

    every = config.every_var_name
    aligned = []
    for li, (nm, od) in enumerate(chunks):
        items = []
        for k, v in od.items():
            if li != len(chunks) - 1:
                if restore:
                    v = restore_only_symbols(v)
                v = DendroConfiguration.find_and_replace_complex_ders(
                    v, every, 0, config.idx_str)
            items.append((k, derivs_to_symbols(v, config.idx_str)))
        aligned.append((nm, OrderedDict(items)))

    produced = {sym.Symbol(k) for _n, od in aligned for k in od}
    leaves = set()
    for _n, od in aligned:
        for v in od.values():
            leaves |= v.free_symbols
    leaves -= produced

    ir = build_cascade_ir(aligned, leaves, target_L=options.L,
                          smart_split=options.smart_split, cse_prefix=options.cse_prefix,
                          verbose=options.verbose, auto_layers=options.auto,
                          auto_search_order=options.search_order)
    if len(ir.chunks) > 1 and not any(c.n_prior_refs > 0 for c in ir.chunks):
        raise CascadeNamingError(
            "no layer references a prior layer: by-value substitution found none of the "
            "spec's tensors inside the RHS (derivative-form mismatch?)")
    return ir


# ---------------------------------------------------------------------------
# 2./3. body + adapter files
# ---------------------------------------------------------------------------
_STRUCT_MEMBER_RE = re.compile(r"^\s*double\s*\*\s*(\w+)\s*;", re.M)


def struct_members(deriv_struct_text: str) -> set:
    return set(_STRUCT_MEMBER_RE.findall(deriv_struct_text))


def _alias_target(token, in_names, use_advective, staged_names):
    """Run the flat path's own naming passes on one token -> `d.X_x` / `in.X`."""
    from dendrosym import codegen
    t = token
    if not use_advective:
        t = codegen.fold_agrad_to_grad(t)
    t = codegen.rename_deriv_buffers(t, in_names, use_advective)
    t = codegen.apply_deriv_struct(t, in_names, extra_names=staged_names)
    return t


def emit_config_cascade(ir, options, *, config, var_type, deriv_struct_text,
                        in_names, use_advective, staged_names=(),
                        project_name="") -> CascadeParts:
    from dendrosym.cascade.emit import (emit_body, kernel_signature, macro_block,
                                        masked_body, scalar_macro_block, undef_block)
    if staged_names:
        raise NotImplementedError("cascade + staged/intermediate buffers is not supported")

    # the VEC body is width-agnostic (only the macro set differs between AVX2 and
    # AVX-512), so it is emitted ONCE and the wrapper picks the width at compile
    # time. `options.simd` therefore does not affect solver output.
    options = options.replace(simd="avx2")
    sig = kernel_signature(ir, options)
    members = struct_members(deriv_struct_text)
    out_fields = list(config.all_var_names.get(var_type, []))
    in_struct = config.input_struct_name()
    out_struct = config.output_struct_name(var_type)

    # --- header + body (the body is exactly what the engine emits for this IR)
    scalars = list(sig["scalars"])
    header = [
        f"// {project_name or config.project_name} {var_type} RHS via polynomial cascade "
        "-- IR-driven, SIMD-batched (dendrosym.cascade)",
        f"// L = {len(ir.chunks)} chunks; width-agnostic VEC body (AVX2 4-wide / AVX-512 8-wide "
        "chosen at compile time by the wrapper's select file); "
        f"inline_threshold={options.inline_threshold} fma_tree={options.fma_tree} "
        f"global_cse={options.global_cse} L={options.L} auto={options.auto}",
        "// Per-chunk CSE only; named tensors survive as chunk boundaries.",
        "// Wrapper provides: VEC macros, pp, the alias table"
        + (f", and {', '.join(scalars)} as VEC." if scalars else "."),
        "",
    ]
    body = emit_body(ir, options, header="\n".join(header) + "\n", vec=True)

    # --- alias table -------------------------------------------------------
    alias = OrderedDict()
    for arr in sorted(sig["pp_arrays"]):
        t = _alias_target(arr, in_names, use_advective, staged_names)
        if t.startswith("d."):
            if t[2:] not in members:
                raise CascadeNamingError(
                    f"cascade leaf {arr} maps to {t} but the {var_type} deriv struct has "
                    f"no such buffer (members: {len(members)})")
            alias[arr] = f"const double *const {arr} = {t};"
        elif t == arr and arr in in_names:
            alias[arr] = f"const double *const {arr} = {in_struct}.{arr};"
        else:
            raise CascadeNamingError(f"cannot map cascade leaf {arr!r} (-> {t!r}) onto the wrapper")
    for o in sig["outputs"]:
        if not o.endswith("_rhs"):
            raise CascadeNamingError(f"output {o!r} is not a *_rhs name")
        field = o[:-4]
        if field not in out_fields:
            raise CascadeNamingError(f"output {o!r} is not a {var_type} field")
        alias[o] = f"double *const {o} = {out_struct}.{field};"

    # bare scalars: coordinate leaves get lane recipes; everything else is a
    # parameter the wrapper already declares at function scope (same source the
    # flat body reads: parameter_code_<vt>) -- copy it, then VSET per batch.
    prologue = []
    coords = [s for s in scalars if s in COORD_LEAVES]
    params = [s for s in scalars if s not in COORD_LEAVES]
    if coords:
        prologue += [
            "alignas(64) double __cascade_xl[__cascade_W];",
            "for (unsigned int __l = 0; __l < __cascade_W; __l++) "
            "__cascade_xl[__l] = pmin[0] + (__cascade_i0 + __l) * hx;",
        ]
        recipes = {
            "x": "const VEC x = VLOAD(__cascade_xl);",
            "y": "const VEC y = VSET(pmin[1] + j * hy);",
            "z": "const VEC z = VSET(pmin[2] + k * hz);",
            "r_coord": "const VEC r_coord = VSQRT(VADD(VADD(VMUL(x, x), VMUL(y, y)), VMUL(z, z)));",
            "t": "const VEC t = VSET(__cascade_scalar_t);",
        }
        order = [c for c in ("x", "y", "z", "r_coord", "t") if c in coords]
        if "r_coord" in order:
            for c in ("x", "y", "z"):
                if c not in order:
                    order.insert(order.index("r_coord"), c)
        if "t" in order:
            alias["t"] = "const double __cascade_scalar_t = t;"
        prologue += [recipes[c] for c in order]
    for s in params:
        alias[s] = f"const double __cascade_scalar_{s} = {s};"
        prologue.append(f"const VEC {s} = VSET(__cascade_scalar_{s});")

    # --- collision check ---------------------------------------------------
    cascade_names = set(alias) | {f"v_{a}" for a in sig["pp_arrays"]}
    cascade_names |= {f"{b}_{i}" for b, i in sig["idx_consts"]}
    for c in ir.chunks:
        cascade_names |= set(c.outputs)
        cascade_names |= {str(s) for s, _ in c.cse_temps}
    bad = sorted((cascade_names - set(params) - set(coords)) & RESERVED_WRAPPER_NAMES)
    bad += sorted(n for n in cascade_names if n.startswith("__cascade_"))
    if bad:
        raise CascadeNamingError(f"cascade names collide with the wrapper scope: {bad[:10]}")
    dup = [o for c in ir.chunks for o in c.outputs if o in alias and "_rhs" not in o]
    if dup:
        raise CascadeNamingError(f"layer outputs shadow alias names: {dup[:10]}")

    # --- macro sets: one file per ISA + the compile-time selector -------------
    guards = {
        "avx2": ["#if !defined(__AVX2__) || !defined(__FMA__)",
                 '#error "cascade AVX2 macro set needs -mavx2 -mfma (CPU_ARCH in CMakeLists.txt)"',
                 "#endif"],
        "avx512": ["#if !defined(__AVX512F__)",
                   '#error "cascade AVX-512 macro set needs -mavx512f (CPU_ARCH in CMakeLists.txt)"',
                   "#endif"],
    }
    macros_avx2 = "\n".join(["// cascade macro set: avx2 (4-wide) -- file scope"]
                            + guards["avx2"] + macro_block("avx2")) + "\n"
    macros_avx512 = "\n".join(["// cascade macro set: avx512 (8-wide) -- file scope"]
                              + guards["avx512"] + macro_block("avx512")) + "\n"
    macros_scalar = "\n".join(["// cascade width-1 macro set (VEC = double); usable inside a function"]
                              + scalar_macro_block()) + "\n"
    macros_undef = "\n".join(["// cascade macro cleanup"] + undef_block()
                             + ["#undef __cascade_W"]) + "\n"
    fn = cascade_filenames(project_name or config.project_name, var_type)
    select = "\n".join([
        "// cascade kernel selection (compile time). Follows the target ISA -- CPU_ARCH sets",
        "// -march, the compiler defines __AVX512F__ / __AVX2__+__FMA__ -- unless forced by",
        "// the CMake option CASCADE_KERNEL=flat|avx2|avx512 (DENDRO_CASCADE_FORCE_*).",
        "// With no usable SIMD the flat CSE kernel is compiled instead (DENDRO_CASCADE_FLAT).",
        "#undef DENDRO_CASCADE_FLAT",
        "#undef __cascade_W",
        "#if defined(DENDRO_CASCADE_FORCE_FLAT)",
        "#  define DENDRO_CASCADE_FLAT 1",
        "#elif defined(DENDRO_CASCADE_FORCE_AVX512) || "
        "(!defined(DENDRO_CASCADE_FORCE_AVX2) && defined(__AVX512F__))",
        f'#  include "{fn["macros_avx512"]}"',
        "#  define __cascade_W 8u",
        "#elif defined(DENDRO_CASCADE_FORCE_AVX2) || (defined(__AVX2__) && defined(__FMA__))",
        f'#  include "{fn["macros_avx2"]}"',
        "#  define __cascade_W 4u",
        "#else",
        "#  define DENDRO_CASCADE_FLAT 1",
        "#endif",
    ]) + "\n"

    manifest = {
        "bridge_version": CASCADE_BRIDGE_VERSION,
        "project": project_name or config.project_name,
        "var_type": var_type,
        "options": {k: v for k, v in dataclasses.asdict(options).items()},
        "simd": "compile-time", "widths": {"avx2": 4, "avx512": 8},
        "layers": [(c.name, c.n_temps, len(c.outputs), c.n_prior_refs) for c in ir.chunks],
        "pp_arrays": list(sig["pp_arrays"]),
        "idx_consts": [[b, i] for b, i in sig["idx_consts"]],
        "scalars": scalars, "coords": coords, "params": params,
        "outputs": list(sig["outputs"]),
        "alias": {k: v for k, v in alias.items()},
        "in_names": list(in_names), "out_fields": out_fields,
        "struct_members": sorted(members),
        "use_advective": bool(use_advective),
    }
    return CascadeParts(
        body=body,
        body_tail=masked_body(body),
        alias="\n".join(["// cascade alias table (generated): vikr leaf/output names -> Dendro-6 storage"]
                        + list(alias.values())) + "\n",
        prologue="\n".join(["// cascade per-batch prologue (generated)"] + prologue) + "\n",
        select=select, macros_avx2=macros_avx2, macros_avx512=macros_avx512,
        macros_scalar=macros_scalar, macros_undef=macros_undef,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# files + template context
# ---------------------------------------------------------------------------
FILE_KEYS = ("body", "body_tail", "alias", "prologue", "select", "macros_avx2", "macros_avx512",
             "macros_scalar", "macros_undef", "manifest")


def cascade_filenames(prefix: str, var_type: str) -> dict:
    ext = {"manifest": "json"}
    return {k: f"{prefix}_{var_type}_cascade_{k}.{ext.get(k, 'cpp.inc')}" for k in FILE_KEYS}


def write_cascade_files(gencode_dir, prefix: str, var_type: str, parts: CascadeParts) -> dict:
    gencode_dir = Path(gencode_dir)
    files = cascade_filenames(prefix, var_type)
    for k in FILE_KEYS:
        content = getattr(parts, k)
        if k == "manifest":
            content = json.dumps(content, indent=1, sort_keys=True) + "\n"
        (gencode_dir / files[k]).write_text(content)
    return files


def cascade_ctx(files: dict, options) -> dict:
    """The ``ctx[f"{vt}_cascade"]`` dict the rhs.cpp.j2 branch reads."""
    return {"simd": "compile-time", **files}


# ---------------------------------------------------------------------------
# verification harness (gate b): flat body vs cascade wrapper on random inputs
# ---------------------------------------------------------------------------
def emit_verification_harness(manifest_path, gencode_dir, out_cpp, n_points=4099,
                              amp=1e-4, tol=1e-12) -> str:
    """Write a standalone C++ program that evaluates the FLAT rhs_eqns body and
    the cascade wrapper (alias + prologue + body, W-loop with shift-back tail
    and the width-1 fallback) on the same random inputs and reports the max
    relative difference per output. Mirrors vikr's pseudo-verify gate
    (random amplitude 1e-4, tolerance 1e-12). Returns the path written."""
    m = json.loads(Path(manifest_path).read_text())
    gencode_dir = Path(gencode_dir).resolve()
    prefix, vt = m["project"], m["var_type"]
    files = cascade_filenames(prefix, vt)
    flat = gencode_dir / f"{prefix}_{vt}_rhs_eqns.cpp.inc"
    unit = {"alpha", "chi"} | {f for f in m["in_names"] if re.fullmatch(r"gt(00|11|22)", f)}
    idx_bases = {}
    for b, i in m["idx_consts"]:
        idx_bases[b] = max(idx_bases.get(b, 0), i + 1)
    L = []
    L += ["// AUTO-GENERATED by dendrosym.cascade.dendro_bridge.emit_verification_harness",
          "#include <cmath>", "#include <cstdio>", "#include <cstdlib>", "#include <vector>",
          f'#include "{gencode_dir / files["select"]}"',
          "#ifdef DENDRO_CASCADE_FLAT",
          '#error "harness: build with -mavx2 -mfma (and -mavx512f for the 8-wide check)"',
          "#endif", "",
          "static unsigned long long _s = 88172645463325252ULL;",
          "static double urand() { _s ^= _s << 13; _s ^= _s >> 7; _s ^= _s << 17;"
          " return (double)(_s % 2000001) / 1000000.0 - 1.0; }", "",
          "struct in_t { " + " ".join(f"const double *{f};" for f in m["in_names"]) + " };",
          "struct d_t { " + " ".join(f"double *{f};" for f in m["struct_members"]) + " };",
          "struct out_t { " + " ".join(f"double *{f};" for f in m["out_fields"]) + " };", "",
          "int main() {", f"    const int N = {n_points};",
          f"    const double amp = {amp};",
          "    std::vector<std::vector<double>> pool;",
          "    auto fresh = [&](double base) { pool.emplace_back(N); auto &v = pool.back();"
          " for (int i = 0; i < N; i++) v[i] = base + amp * urand(); return v.data(); };"]
    for b, n in sorted(idx_bases.items()):
        vals = ", ".join(f"{0.1 * (i + 1):.3f}" for i in range(n))
        L.append(f"    const double {b}[{n}] = {{{vals}}};")
    for s in m["params"]:
        L.append(f"    const double {s} = 2.0;")
    if m["coords"]:
        L += ["    const double pmin[3] = {0.1, 0.2, 0.3}; const double hx = 0.01, hy = 0.01, hz = 0.01;",
              "    const unsigned int j = 1, k = 2; const double t = 0.5;"]
    L += ["    in_t in; d_t d; out_t out, out_flat, out_casc;"]
    L += [f"    in.{f} = fresh({1.0 if f in unit else 0.0});" for f in m["in_names"]]
    L += [f"    d.{f} = fresh(0.0);" for f in m["struct_members"]]
    L += [f"    out_flat.{f} = fresh(0.0); out_casc.{f} = fresh(0.0);" for f in m["out_fields"]]
    # flat
    L += ["    out = out_flat;", "    for (int pp = 0; pp < N; pp++) {"]
    if m["coords"]:
        L += ["        const double x = pmin[0] + pp * hx; const double y = pmin[1] + j * hy;",
              "        const double z = pmin[2] + k * hz; const double r_coord = sqrt(x*x + y*y + z*z);"]
    L += [f'#include "{flat}"', "    }"]
    # cascade (same shape as the rhs.cpp.j2 branch, one row of N points)
    L += ["    out = out_casc;", "    {",
          f'#include "{gencode_dir / files["alias"]}"',
          "        const unsigned int __cascade_i_lo = 0, __cascade_i_hi = N;",
          "        for (unsigned int i = __cascade_i_lo; i < __cascade_i_hi; i += __cascade_W) {",
          "            const unsigned int __cascade_nvalid = (__cascade_i_hi - i < __cascade_W)"
          " ? (__cascade_i_hi - i) : __cascade_W;",
          "            const bool __cascade_shift = (__cascade_nvalid < __cascade_W) &&"
          " (__cascade_i_hi >= __cascade_i_lo + __cascade_W);",
          "            const unsigned int __cascade_i0 = __cascade_shift ? (__cascade_i_hi - __cascade_W) : i;"
          " const unsigned int pp = __cascade_i0;",
          f'#include "{gencode_dir / files["prologue"]}"',
          "            if (__cascade_nvalid == __cascade_W || __cascade_shift) {",
          f'#include "{gencode_dir / files["body"]}"',
          "            } else {",
          "                const auto __cascade_mask = VMASK(__cascade_nvalid);",
          f'#include "{gencode_dir / files["body_tail"]}"',
          "            }",
          "        }",
          f'#include "{gencode_dir / files["macros_undef"]}"',
          "    }"]
    # compare
    L += ["    double worst = 0; const char *worst_name = \"\";",
          f"    const double tol = {tol};"]
    for f in m["out_fields"]:
        L += [f"    {{ double m = 0, s = 1; for (int pp = 0; pp < N; pp++) {{"
              f" s = fmax(s, fabs(out_flat.{f}[pp]));"
              f" m = fmax(m, fabs(out_flat.{f}[pp] - out_casc.{f}[pp])); }}"
              f" printf(\"%-12s max|flat-cascade| = %.3e  (scale %.3e)\\n\", \"{f}\", m / s, s);"
              f" if (m / s > worst) {{ worst = m / s; worst_name = \"{f}\"; }} }}"]
    L += ["    printf(\"WORST %s %.3e (tol %.1e) -> %s\\n\", worst_name, worst, tol,"
          " worst <= tol ? \"PASS\" : \"FAIL\");",
          "    return worst <= tol ? 0 : 1;", "}", ""]
    Path(out_cpp).write_text("\n".join(L))
    return str(out_cpp)


def _main(argv=None):
    import argparse
    import subprocess
    import sys
    ap = argparse.ArgumentParser(prog="python -m dendrosym.cascade.dendro_bridge",
                                 description="gate (b): flat-vs-cascade verification harness")
    ap.add_argument("--harness", required=True, metavar="MANIFEST_JSON")
    ap.add_argument("--out-dir", default=None, help="default: <gencode>/.cascade_check")
    ap.add_argument("--run", action="store_true", help="compile (-O3 and -O0) and run")
    ap.add_argument("--cxx", default="g++")
    ns = ap.parse_args(argv)
    manifest = Path(ns.harness).resolve()
    gencode_dir = manifest.parent
    out_dir = Path(ns.out_dir) if ns.out_dir else gencode_dir / ".cascade_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    m = json.loads(manifest.read_text())
    cpp = out_dir / f"{m['project']}_{m['var_type']}_cascade_check.cpp"
    emit_verification_harness(manifest, gencode_dir, cpp)
    print(f"wrote {cpp}")
    if not ns.run:
        return 0
    # every width the host can run: AVX2 always (-O3 and -O0), AVX-512 when the CPU has it
    variants = [("avx2", "-O3", ["-mavx2", "-mfma", "-DDENDRO_CASCADE_FORCE_AVX2"]),
                ("avx2", "-O0", ["-mavx2", "-mfma", "-DDENDRO_CASCADE_FORCE_AVX2"])]
    try:
        host_avx512 = "avx512f" in Path("/proc/cpuinfo").read_text()
    except OSError:
        host_avx512 = False
    if host_avx512:
        variants.append(("avx512", "-O3", ["-mavx512f", "-mavx2", "-mfma"]))
    rc = 0
    for isa, opt, flags in variants:
        exe = out_dir / f"check-{isa}{opt}"
        cmd = [ns.cxx, opt, *flags, "-std=c++17", str(cpp), "-o", str(exe)]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1], f"[{isa} {opt}]")
        rc |= r.returncode
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_main())
