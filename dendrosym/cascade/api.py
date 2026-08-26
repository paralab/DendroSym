"""cascade_api -- the single-import entry point for the polynomial cascade.

The intended workflow is: write YOUR OWN python file, declare your system's
named objects in evaluation order, import this module, run your file.

    from collections import OrderedDict
    import sympy as sym
    from dendrosym.cascade import compile_system, CascadeOptions

    a, x = sym.symbols("a x")
    inv, g = sym.symbols("inv g")          # names for your intermediates
    specs = [
        ("inv", OrderedDict(inv=1/a)),
        ("g",   OrderedDict(g=x*x + x)),
        ("out", OrderedDict(out=g*inv)),   # reference earlier outputs BY SYMBOL
    ]
    code, ir = compile_system(specs, {a, x}, out="my_kernel.cpp")
    # every paper knob, one call:
    code, ir = compile_system(specs, {a, x},
                              options=CascadeOptions(simd="avx512", L=7, fma_tree=True))

Conventions that matter:
  * Objects are declared in the order you would derive them at a blackboard.
    Grouping is optional -- with auto layering only the ORDER is used; the
    layer boundaries are chosen by an exact DP over cut positions (see
    dendrosym.cascade.autolayer).
  * Downstream expressions must reference earlier outputs BY SYMBOL (a bare
    sym.Symbol carrying the output's name). Pasting the full expression tree
    ("by value") is never wrong but breaks subtree matching under sympy's
    Add-flattening: the machinery silently recomputes and the liveness model
    smears.
  * Generated code is numerically identical to direct evaluation of your
    expressions (exact substitution; per-layer CSE only).
  * Naming drives classification: `name[pp]` -> per-point array, `name[N]` ->
    indexed constant, bare `name` -> scalar the wrapper provides. Outputs that
    should be array stores contain `_rhs` or end in `_out`.

Emitted kernels are sympy-version dependent (per-chunk CSE); the pinned
environment is requirements-cascade.txt (sympy 1.13.3, PYTHONHASHSEED=0).
"""
import os
import warnings

from dendrosym.cascade.builder import build_cascade_ir
from dendrosym.cascade.options import CascadeOptions

PINNED_SYMPY = "1.13.3"


def warn_if_unpinned():
    """Warn (once) when sympy differs from the pinned version: kernels still
    gate at machine precision but will not be byte-reproducible."""
    import sympy
    if sympy.__version__ != PINNED_SYMPY and not os.environ.get("CASCADE_NO_PIN_WARNING"):
        warnings.warn(
            f"dendrosym.cascade: sympy {sympy.__version__} != pinned {PINNED_SYMPY}; "
            "emitted kernels will differ byte-wise from the validated ones "
            "(see requirements-cascade.txt). Set CASCADE_NO_PIN_WARNING=1 to silence.",
            stacklevel=2)


def build(specs, leaves, options=None, **kw):
    """Spec -> CascadeResult under CascadeOptions (build knobs only)."""
    o = options or CascadeOptions(**kw)
    return build_cascade_ir(list(specs), set(leaves), target_L=o.L,
                            smart_split=o.smart_split, cse_prefix=o.cse_prefix,
                            verbose=o.verbose, auto_layers=o.auto,
                            auto_search_order=o.search_order)


def compile_system(specs, leaves, L=None, auto=True, cse_prefix="CASC_",
                   out=None, verbose=False, *, options=None):
    """Declared objects -> layered, CSE'd C++ kernel.

    Parameters
    ----------
    specs : list[(name, OrderedDict[str, sympy.Expr])]
        Named objects in evaluation order (see module docstring).
    leaves : set[sympy.Symbol]
        The inputs: state variables, derivative placeholders, parameters.
    L, auto, cse_prefix, out, verbose
        The original quick-start knobs (kept byte-compatible: with no
        `options` this emits exactly what it always did -- the plain unrolled
        scalar body, auto-layered by default).
    options : CascadeOptions, optional
        When given, it wins over the positional knobs and selects the emitter:
        the result is a complete, self-contained kernel translation unit
        (emit_standalone_kernel: macro set + `static inline void fn(int N,
        ...)`) at options.simd width, honouring inline_threshold / fma_tree /
        global_cse / lazy_prologue / L / auto. Outputs are the objects whose
        names contain `_rhs` or end in `_out`. `options.out` writes the file.
        For a BODY to `#include` in your own loop use build() + emit_body().

    Returns (code, ir): the C++ text and the CascadeResult (layer names,
    per-layer temps/ops, outputs) for inspection or re-emission.
    """
    if options is None:
        ir = build_cascade_ir(list(specs), set(leaves), target_L=L,
                              cse_prefix=cse_prefix, verbose=verbose,
                              auto_layers=auto)
        code = ir.emit_cpp_unrolled(dendro_var_style=False, inline_threshold=0,
                                    short_names=False)
    else:
        from dendrosym.cascade.emit import emit_standalone_kernel
        ir = build(specs, leaves, options)
        if options.emit_style == "tensor-loop":
            raise NotImplementedError(
                "emit_style='tensor-loop' needs a NamingDialect: call "
                "ir.emit_cpp_tensor_loop(dialect) directly (see dendrosym.cascade.dialect)")
        code = emit_standalone_kernel(ir, options)
        out = options.out or out
        verbose = options.verbose or verbose
    if out:
        with open(out, "w") as f:
            f.write(code)
        if verbose:
            print(f"compile_system: wrote {len(code.splitlines())} lines "
                  f"to {out}")
    return code, ir


def predict(specs, leaves, options=None):
    """Cost model WITHOUT emitting a file: build the IR and measure the
    source-level peak of simultaneously live compute values (the paper's
    predictor of register pressure / spills). Returns a dict:
    peak_live, temps, layers, prior_refs (per layer), n_outputs."""
    import tempfile
    from dendrosym.cascade.liveness_viz import parse_kernel, source_liveness

    ir = build(specs, leaves, options or CascadeOptions())
    code = ir.emit_cpp_unrolled(dendro_var_style=True, inline_threshold=0,
                                short_names=False)
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        stmts, roots, labels = parse_kernel(path)
        lv = source_liveness(stmts)
    finally:
        os.unlink(path)
    return {"peak_live": int(lv["peak"]), "trace": list(lv["trace"]), "temps": sum(c.n_temps for c in ir.chunks),
            "layers": len(ir.chunks), "prior_refs": [c.n_prior_refs for c in ir.chunks],
            "n_outputs": sum(len(c.outputs) for c in ir.chunks), "ir": ir}


def report(ir):
    """One-screen summary of a compiled cascade: layers, widths, temps."""
    print(f"{len(ir.chunks)} layers, "
          f"{sum(c.n_temps for c in ir.chunks)} CSE temps, "
          f"{sum(len(c.outputs) for c in ir.chunks)} named outputs")
    for i, c in enumerate(ir.chunks):
        print(f"  L{i+1}: {c.name:40s} {len(c.outputs):3d} outputs, "
              f"{c.n_temps:4d} temps, reads {c.n_prior_refs} prior")
