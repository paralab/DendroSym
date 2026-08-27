"""__init__.py

This is the initialization file for the DendroSym Python package.
It helps us make sure that everything is ready to roll.

Submodules are loaded lazily (PEP 562 ``__getattr__``): ``import dendrosym`` is
cheap and pulls in nothing but the standard library, and ``dendrosym.nr``,
``dendrosym.dtypes``, ``dendrosym.NRConfig`` ... resolve on first use exactly as
they did when this file imported everything eagerly. The point is that the
polynomial-cascade generator (``dendrosym.cascade``) depends only on sympy /
numpy / networkx and must import in an environment without the solver
toolkit's jinja2 / tomlkit / matplotlib / dill / tqdm.
"""
import importlib
import sys as _sys

# every submodule the eager __init__ used to import (same attribute names)
_SUBMODULES = (
    "dtypes",          # the datatypes that includes all of the sympy pieces
    "nr",              # numerical relativity functions
    "codegen",         # code generation
    "params",          # parameter information
    "memoryManager",   # the basic memory manager
    "nxgraph",         # nxgraph generation
    "refEl",           # reference element information
    "utils",
    "derivs",
    "helpers",
    "general_configs",
    "nr_configs",
    "code_printer",    # the code printer
    "project_generator",  # project generator (template-based, replaces cog)
    "gr_symbols",
    "cascade",         # polynomial-cascade code generation (sympy-only)
)
# names re-exported from submodules
_ATTRS = {
    "DendroConfiguration": "general_configs",   # base configuration class
    "NRConfig": "nr_configs",                   # the numerical relativity class
    "DendroProjectGenerator": "project_generator",
}
# not imported (kept from the eager file's notes):
#   sympy_cachesim -- does not work on some machines
#   gw             -- not currently supported
#   graph_coarsening


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _ATTRS:
        mod = importlib.import_module(f"{__name__}.{_ATTRS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES) | set(_ATTRS))


def run(config, default_output_dir=None, argv=None):
    """One-call entry point: parse argv, build the generator, and emit.

    Usage:
        if __name__ == "__main__":
            dendrosym.run(dendroConfigs)

    Recognized argv flags:
        --skip-gencode    skip the expensive CSE/equation-printing pass
        --gencode-only    only emit gencode .cpp.inc files
        --profile / --no-profile
                          print an MPI-reduced timer table (rhs body, derivatives,
                          KO, BCs, zip/unzip) at the end of the run
        --cascade / --no-cascade
                          emit (or suppress) the polynomial-cascade SIMD kernel for
                          every var_type with a registered spec
                          (config.set_cascade_spec_function); flags win over the
                          options the config script registered
        --cascade-<knob>  override one CascadeOptions field, e.g. --cascade-simd
                          avx512 --cascade-L 7 --cascade-no-fma-tree (see
                          dendrosym.cascade.CascadeOptions; --help lists them)
        <positional>      output directory (defaults to caller's __file__ dir)
    """
    import argparse
    import os

    from .project_generator import DendroProjectGenerator

    if argv is None:
        argv = _sys.argv[1:]

    ap = argparse.ArgumentParser(
        prog=os.path.basename(_sys.argv[0]) or "dendrosym.run",
        description=f"generate the {config.project_name} solver")
    ap.add_argument("output_dir", nargs="?", default=None)
    ap.add_argument("--skip-gencode", action="store_true")
    ap.add_argument("--gencode-only", action="store_true")
    ap.add_argument("--profile", dest="profile", action="store_true", default=None,
                    help="emit the end-of-run timer table (config.enable_profiling)")
    ap.add_argument("--no-profile", dest="profile", action="store_false")
    ap.add_argument("--cascade", dest="cascade", action="store_true", default=None,
                    help="emit the polynomial-cascade kernel (registered specs)")
    ap.add_argument("--no-cascade", dest="cascade", action="store_false")
    from dendrosym.cascade.options import CascadeOptions
    CascadeOptions.add_argparse_args(ap, prefix="cascade-")
    ns, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"dendrosym.run: ignoring unrecognized arguments {unknown}", file=_sys.stderr)

    skip = ns.skip_gencode
    if ns.profile is not None:
        config.enable_profiling = ns.profile
    gencode_only = ns.gencode_only

    # cascade overrides: CLI flags layer over whatever the config script registered
    if hasattr(config, "override_cascade"):
        specs = getattr(config, "stored_cascade_specs", {})
        for vt, (fn, opts) in list(specs.items()):
            new = CascadeOptions.from_namespace(ns, prefix="cascade-", base=opts)
            if ns.cascade is not None:
                new = new.replace(enabled=ns.cascade)
            specs[vt] = (fn, new)
        if ns.cascade and not specs:
            print("dendrosym.run: --cascade given but the config registered no cascade "
                  "spec (set_cascade_spec_function)", file=_sys.stderr)

    if ns.output_dir:
        output_dir = ns.output_dir
    elif default_output_dir is not None:
        output_dir = default_output_dir
    else:
        # caller's __file__ directory
        import inspect
        caller_file = inspect.stack()[1].filename
        output_dir = os.path.dirname(os.path.abspath(caller_file))

    print(f"Generating {config.project_name} solver into: {output_dir}",
          file=_sys.stderr)
    gen = DendroProjectGenerator(config)
    gen.generate(output_dir, skip_gencode=skip, gencode_only=gencode_only)


__version__ = "0.0.1"
