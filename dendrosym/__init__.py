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
        <positional>      output directory (defaults to caller's __file__ dir)
    """
    import os
    import sys

    if argv is None:
        argv = sys.argv[1:]

    skip = "--skip-gencode" in argv
    gencode_only = "--gencode-only" in argv
    positional = [a for a in argv if not a.startswith("--")]

    if positional:
        output_dir = positional[0]
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
