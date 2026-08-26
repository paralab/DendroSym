"""cascade_api -- the single-import entry point for the polynomial cascade.

The intended workflow is: write YOUR OWN python file, declare your system's
named objects in evaluation order, import this module, run your file.

    from collections import OrderedDict
    import sympy as sym
    from dendrosym.cascade.api import compile_system

    a, x = sym.symbols("a x")
    inv, g = sym.symbols("inv g")          # names for your intermediates
    specs = [
        ("inv", OrderedDict(inv=1/a)),
        ("g",   OrderedDict(g=x*x + x)),
        ("out", OrderedDict(out=g*inv)),   # reference earlier outputs BY SYMBOL
    ]
    code, ir = compile_system(specs, {a, x}, out="my_kernel.cpp")

Conventions that matter:
  * Objects are declared in the order you would derive them at a blackboard.
    Grouping is optional -- with auto layering (the default) only the ORDER
    is used; the layer boundaries are chosen by an exact DP over cut
    positions (see cascade_autolayer).
  * Downstream expressions must reference earlier outputs BY SYMBOL (a bare
    sym.Symbol carrying the output's name). Pasting the full expression tree
    ("by value") is never wrong but breaks subtree matching under sympy's
    Add-flattening: the machinery silently recomputes and the liveness model
    smears.
  * Generated code is numerically identical to direct evaluation of your
    expressions (exact substitution; per-layer CSE only).
"""
from collections import OrderedDict

from dendrosym.cascade.builder import build_cascade_ir


def compile_system(specs, leaves, L=None, auto=True, cse_prefix="CASC_",
                   out=None, verbose=False):
    """Declared objects -> layered, CSE'd, unrolled C++ kernel body.

    Parameters
    ----------
    specs : list[(name, OrderedDict[str, sympy.Expr])]
        Named objects in evaluation order (see module docstring).
    leaves : set[sympy.Symbol]
        The inputs: state variables, derivative placeholders, parameters.
    L : int or None
        Layer count. With auto=True (default), None lets the DP choose the
        depth as well as the boundaries. With auto=False, None means "as
        declared" and L invokes the collapse/split transforms instead.
    auto : bool
        Choose layer boundaries automatically from the declared order.
    out : str or None
        If given, also write the emitted C++ there.

    Returns (code, ir): the C++ body string and the CascadeResult (layer
    names, per-layer temps/ops, outputs) for inspection or re-emission.
    """
    ir = build_cascade_ir(list(specs), set(leaves), target_L=L,
                          cse_prefix=cse_prefix, verbose=verbose,
                          auto_layers=auto)
    code = ir.emit_cpp_unrolled(dendro_var_style=False, inline_threshold=0,
                                short_names=False)
    if out:
        with open(out, "w") as f:
            f.write(code)
        if verbose:
            print(f"compile_system: wrote {len(code.splitlines())} lines "
                  f"to {out}")
    return code, ir


def report(ir):
    """One-screen summary of a compiled cascade: layers, widths, temps."""
    print(f"{len(ir.chunks)} layers, "
          f"{sum(c.n_temps for c in ir.chunks)} CSE temps, "
          f"{sum(len(c.outputs) for c in ir.chunks)} named outputs")
    for i, c in enumerate(ir.chunks):
        print(f"  L{i+1}: {c.name:40s} {len(c.outputs):3d} outputs, "
              f"{c.n_temps:4d} temps, reads {c.n_prior_refs} prior")
