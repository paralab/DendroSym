"""__init__.py

This is the initialization file for the DendroSym Python package.
It helps us make sure that everything is ready to roll
"""

# the datatypes that includes all of the sympy pieces
from . import dtypes

# numerical relativity functions
from . import nr

# code generation
from . import codegen

# parameter information
from . import params

# ==== MISC. FUNCTIONS AND OPERATIONS ====
# the basic memory manager
from . import memoryManager

# nxgraph generation
from . import nxgraph

# reference element information
from . import refEl

# sympy cache simulations
# from . import sympy_cachesim
# TODO: cachesim currently does not work on some machines

from . import utils

from . import params

from . import derivs

from . import helpers

# ====== DENDRO CONFIGURATION ======
# base configuration class
from .general_configs import DendroConfiguration

# and then the numerical relativity class
from .nr_configs import NRConfig

# the code printer also needs to be imported
from . import code_printer

# from .dendro_generator import DendroGenerator

# project generator (template-based, replaces cog)
from .project_generator import DendroProjectGenerator


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
    import sys

    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]) or "dendrosym.run",
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
        print(f"dendrosym.run: ignoring unrecognized arguments {unknown}", file=sys.stderr)

    skip = ns.skip_gencode
    if ns.profile is not None:
        config.enable_profiling = ns.profile
    gencode_only = ns.gencode_only

    # cascade overrides: CLI flags layer over whatever the config script registered
    if hasattr(config, "override_cascade"):
        specs = getattr(config, "stored_cascade_specs", {})
        changes = {}
        for vt, (fn, opts) in list(specs.items()):
            new = CascadeOptions.from_namespace(ns, prefix="cascade-", base=opts)
            if ns.cascade is not None:
                new = new.replace(enabled=ns.cascade)
            specs[vt] = (fn, new)
        if ns.cascade and not specs:
            print("dendrosym.run: --cascade given but the config registered no cascade "
                  "spec (set_cascade_spec_function)", file=sys.stderr)

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
          file=sys.stderr)
    gen = DendroProjectGenerator(config)
    gen.generate(output_dir, skip_gencode=skip, gencode_only=gencode_only)

# TODO: the package that is used with gw is not currently supported!
# from . import gw


# TODO: quaternion and cachesym are not working on MaryLou currently


# from . import graph_coarsening

__version__ = "0.0.1"
